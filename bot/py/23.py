from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import time
import hashlib
from typing import Optional, Dict, Any, List, Set
from pydantic import BaseModel, Field
import httpx
from pathlib import Path
import edge_tts
import os
import websockets
from functools import lru_cache

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "granite3.2-vision"
DRIVER_WEBSOCKET_URL = "ws://192.168.137.181:8766"
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "en-US-JennyNeural"
DELAY_SECONDS = 0.05  # Reduced from 0.1 for faster response
MAX_SPEED = 0.4
MAX_STEERING_INPUT = 2.0
COMMAND_QUEUE_SIZE = 3  # Maximum number of commands to queue
IMAGE_CACHE_TIME = 0.2  # Time in seconds to cache images
OLLAMA_TIMEOUT = 10.0  # Reduced timeout for Ollama API calls

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "WEBSOCKET"],
    allow_headers=["*"],
)

# 정적 파일 디렉토리 확인 및 생성
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True)
    logger.info(f"Created static directory: {STATIC_DIR}")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Pydantic Models ---
class OllamaRequest(BaseModel):
    prompt: str = Field(..., description="The user's text prompt.")
    delay: float = Field(DELAY_SECONDS, description="Delay between actions in seconds.")


# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # HTML 파일 경로 확인
    index_path = STATIC_DIR / "index_0325_3.html"
    if not index_path.exists():
        # 파일이 없으면 기본 HTML 반환
        logger.warning(f"index_0325_3.html not found at {index_path}")
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>JetBot Control</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
            <h1>JetBot Control Interface</h1>
            <p>HTML file not found. Please place index_0325_2.html in the static directory.</p>
        </body>
        </html>
        """)

    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# --- TTS Function ---
# Cache TTS results to avoid regenerating the same audio
@lru_cache(maxsize=32)
async def generate_tts_cached(text_hash: str, text: str) -> str:
    try:
        if not text or text.isspace():
            text = "Processing..."
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = f"temp_tts_{int(time.time())}.mp3"
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)
        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return ""


async def generate_tts(text: str) -> str:
    # Hash the text to use as a cache key
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return await generate_tts_cached(text_hash, text)


# --- Ollama Interaction ---
# Simplified prompt for faster processing
OLLAMA_PROMPT = """
You are an AI controlling a Professional Driver vehicle. Analyze the image and provide commands to navigate safely.
Generate actionable commands in JSON format using: 'forward', 'backward', 'left', 'right', 'stop', 'u_turn'.
Include 'speed' (0.0-0.4) and 'steering' (-2.0-2.0) parameters.
Add a brief 'tts' field explaining the action.
Prioritize safety and avoid obstacles.
Use this format:
{
  "commands": [
    {"command": "command_name", "parameters": {"speed": float, "steering": float}, "tts": "spoken feedback"}
  ],
  "description": "brief scene description"
}
"""

# Cache Ollama responses to avoid redundant API calls
ollama_cache = {}
ollama_cache_time = {}


async def query_ollama(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    # Generate a cache key based on the image data
    if image_data:
        image_hash = hashlib.md5(image_data.encode()).hexdigest()
        current_time = time.time()

        # Check if we have a recent cached response
        if image_hash in ollama_cache and current_time - ollama_cache_time[image_hash] < IMAGE_CACHE_TIME:
            return ollama_cache[image_hash]
    else:
        image_hash = None

    data = {
        "model": MODEL_NAME,
        "prompt": OLLAMA_PROMPT + prompt,
        "images": [image_data] if image_data else [],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3, "top_p": 0.95, "num_predict": 256},  # Reduced for faster inference
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()
            result = response.json()

            try:
                parsed_response = json.loads(result.get("response", "{}"))
                commands = parsed_response.get("commands", [])
                description = parsed_response.get("description", "No description provided.")

                if not commands:
                    raise ValueError("No valid commands returned")

                result_data = {"commands": commands, "description": description}

                # Cache the result if we have an image hash
                if image_hash:
                    ollama_cache[image_hash] = result_data
                    ollama_cache_time[image_hash] = time.time()

                return result_data
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, response: {result.get('response')}")
                raise

    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return {
            "commands": [
                {"command": "stop", "parameters": {"speed": 0.0, "steering": 0.0}, "tts": "An error occurred."}],
            "description": f"Error: {str(e)}"
        }


# --- WebSocket Connections ---
client_websockets: List[WebSocket] = []
driver_websocket: Optional[websockets.WebSocketClientProtocol] = None
current_image_base64: Optional[str] = None
autonomous_task: Optional[asyncio.Task] = None
is_autonomous_running: bool = False
command_queue: List[Dict[str, Any]] = []
last_broadcast_time = 0
last_image_received_time = 0


async def connect_to_driver():
    global driver_websocket, current_image_base64, last_image_received_time

    while True:
        try:
            logger.info(f"Connecting to driver at {DRIVER_WEBSOCKET_URL}")
            async with websockets.connect(
                    DRIVER_WEBSOCKET_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=10_000_000  # 10MB 최대 메시지 크기 설정
            ) as websocket:
                driver_websocket = websocket
                logger.info("Connected to Driver WebSocket")

                # Start the command processor
                command_processor_task = asyncio.create_task(process_command_queue())

                try:
                    while True:
                        try:
                            data = await websocket.recv()
                            message = json.loads(data)

                            if "image" in message:
                                current_image_base64 = message["image"]
                                last_image_received_time = time.time()

                                # 디버깅을 위한 로그 추가
                                logger.debug(f"Received image: {len(current_image_base64)} bytes")

                                # Only broadcast images at a reasonable rate
                                current_time = time.time()
                                if current_time - last_broadcast_time > 1 / 10:  # 10 FPS max
                                    await broadcast_to_clients({"image": current_image_base64})
                                    last_broadcast_time = current_time
                        except json.JSONDecodeError:
                            logger.error("Failed to parse JSON from driver")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Driver WebSocket connection closed")
                finally:
                    command_processor_task.cancel()
                    driver_websocket = None
        except Exception as e:
            logger.error(f"Driver connection failed: {e}")
            driver_websocket = None
            await asyncio.sleep(5)


async def process_command_queue():
    global command_queue
    while True:
        try:
            if command_queue and driver_websocket:
                command_data = command_queue.pop(0)
                await driver_websocket.send(json.dumps(command_data))
                # Small delay between commands
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Command queue processing error: {e}")
            await asyncio.sleep(0.1)


async def broadcast_to_clients(data: Dict[str, Any]):
    disconnected_clients = []
    for ws in client_websockets:
        try:
            await ws.send_text(json.dumps(data))
        except WebSocketDisconnect:
            disconnected_clients.append(ws)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            disconnected_clients.append(ws)

    for ws in disconnected_clients:
        if ws in client_websockets:
            client_websockets.remove(ws)


@app.websocket("/ws/client")
async def client_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client WebSocket connected")
    client_websockets.append(websocket)
    try:
        # Send the current image immediately if available
        if current_image_base64:
            logger.info("Sending initial image to new client")
            await websocket.send_text(json.dumps({"image": current_image_base64}))

        # 이미지 수신 상태 확인
        current_time = time.time()
        if last_image_received_time > 0 and current_time - last_image_received_time > 10:
            await websocket.send_text(json.dumps({
                "response": "Warning: No recent images from JetBot. Check JetBot connection."
            }))

        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                command = message.get("command", "none")
                parameters = message.get("parameters", {})
                # Process commands in background to avoid blocking
                asyncio.create_task(process_command(command, parameters))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
        if websocket in client_websockets:
            client_websockets.remove(websocket)


def add_to_command_queue(command_data: Dict[str, Any]):
    global command_queue
    # Limit queue size to prevent memory issues
    if len(command_queue) < COMMAND_QUEUE_SIZE:
        command_queue.append(command_data)
    else:
        # Replace the oldest command with the new one
        command_queue.pop(0)
        command_queue.append(command_data)


async def process_command(command: str, parameters: Dict[str, Any]):
    global autonomous_task, is_autonomous_running
    prompt = parameters.get("text", f"Execute {command}")

    if command in ["forward", "backward", "left", "right", "stop", "cruise"]:
        if driver_websocket:
            command_message = {
                "command": command,
                "parameters": {"speed": min(0.2, MAX_SPEED), "steering": 0.0}
            }
            add_to_command_queue(command_message)

            tts_text = f"{command.capitalize()} executed."
            encoded_audio = await generate_tts(tts_text)
            await broadcast_to_clients({
                "response": tts_text,
                "driver_command": command,
                "audio": "data:audio/mp3;base64," + encoded_audio
            })
        else:
            await broadcast_to_clients({"response": "Driver not connected!", "driver_command": "none"})

    elif command == "describe":
        if not current_image_base64:
            await broadcast_to_clients({"response": "No image available!", "driver_command": "none"})
            return

        # Start TTS generation in parallel with Ollama query
        description_task = asyncio.create_task(query_ollama(prompt, current_image_base64))

        # Wait for Ollama response
        ollama_response = await description_task
        description = ollama_response.get("description", "No description.")

        # Generate TTS for the description
        encoded_audio = await generate_tts(description)

        await broadcast_to_clients({
            "response": description,
            "driver_command": "none",
            "audio": "data:audio/mp3;base64," + encoded_audio,
            "description": description
        })

    elif command == "custom":
        if not current_image_base64:
            await broadcast_to_clients({"response": "No image available!", "driver_command": "none"})
            return

        ollama_response = await query_ollama(prompt, current_image_base64)
        commands = ollama_response.get("commands", [])
        description = ollama_response.get("description", "No description.")

        # Process all commands in parallel
        tts_texts = []
        for cmd in commands:
            driver_command = cmd.get("command", "none")
            cmd_params = cmd.get("parameters", {"speed": 0.2, "steering": 0.0})
            cmd_params["speed"] = min(cmd_params.get("speed", 0.2), MAX_SPEED)

            # Normalize steering values
            raw_steering = cmd_params.get("steering", 0.0)
            cmd_params["steering"] = max(min(raw_steering, 1.0), -1.0)

            # Adjust speed for turns
            if driver_command == "u_turn" or abs(raw_steering) > 1.0:
                cmd_params["speed"] = min(cmd_params["speed"], 0.1)

            tts_text = cmd.get("tts", f"Executing {driver_command}.")
            tts_texts.append(tts_text)

            if driver_websocket and driver_command != "none":
                command_message = {
                    "command": driver_command,
                    "parameters": cmd_params
                }
                add_to_command_queue(command_message)

        # Combine TTS texts and generate audio once
        combined_tts = " ".join(tts_texts)
        encoded_audio = await generate_tts(combined_tts)

        await broadcast_to_clients({
            "response": combined_tts,
            "driver_command": [cmd.get("command", "none") for cmd in commands],
            "audio": "data:audio/mp3;base64," + encoded_audio,
            "description": description
        })

    elif command == "autonomous":
        mode = parameters.get("mode", "off")
        if mode == "on" and not is_autonomous_running:
            is_autonomous_running = True
            autonomous_task = asyncio.create_task(autonomous_control(OllamaRequest(prompt=prompt, delay=DELAY_SECONDS)))
            await broadcast_to_clients({"response": "Autonomous mode started.", "driver_command": "none"})
        elif mode == "off" and is_autonomous_running:
            is_autonomous_running = False
            if autonomous_task:
                autonomous_task.cancel()
                try:
                    await autonomous_task
                except asyncio.CancelledError:
                    pass
                autonomous_task = None
            await broadcast_to_clients({"response": "Autonomous mode stopped.", "driver_command": "stop"})
            if driver_websocket:
                add_to_command_queue({"command": "stop", "parameters": {"speed": 0.0, "steering": 0.0}})


# --- Autonomous Control Loop ---
async def autonomous_control(request_data: OllamaRequest):
    global is_autonomous_running
    last_image_hash = None
    last_command_time = 0

    while is_autonomous_running:
        current_time = time.time()

        # Rate limit autonomous control loop
        if current_time - last_command_time < request_data.delay:
            await asyncio.sleep(0.01)
            continue

        last_command_time = current_time

        if not current_image_base64:
            await asyncio.sleep(request_data.delay)
            continue

        # Hash the current image to avoid processing duplicate frames
        current_hash = hashlib.md5(current_image_base64.encode()).hexdigest()
        if current_hash == last_image_hash:
            await asyncio.sleep(request_data.delay)
            continue
        last_image_hash = current_hash

        # Query Ollama for commands
        ollama_response = await query_ollama(request_data.prompt, current_image_base64)
        commands = ollama_response.get("commands", [])

        if not commands:
            await asyncio.sleep(request_data.delay)
            continue

        # Process commands in parallel
        tts_texts = []
        for cmd in commands:
            driver_command = cmd.get("command", "none")
            if driver_command == "none":
                continue

            cmd_params = cmd.get("parameters", {"speed": 0.2, "steering": 0.0})
            cmd_params["speed"] = min(cmd_params.get("speed", 0.2), MAX_SPEED)

            # Normalize steering values
            raw_steering = cmd_params.get("steering", 0.0)
            cmd_params["steering"] = max(min(raw_steering, 1.0), -1.0)

            # Adjust speed for turns
            if driver_command == "u_turn" or abs(raw_steering) > 1.0:
                cmd_params["speed"] = min(cmd_params["speed"], 0.1)

            tts_text = cmd.get("tts", f"Executing {driver_command}.")
            tts_texts.append(tts_text)

            if driver_websocket:
                command_message = {
                    "command": driver_command,
                    "parameters": cmd_params
                }
                add_to_command_queue(command_message)

        # Generate TTS for all commands at once
        if tts_texts:
            combined_tts = " ".join(tts_texts[:3])  # Limit to first 3 commands
            encoded_audio = await generate_tts(combined_tts)

            await broadcast_to_clients({
                "driver_command": [cmd.get("command", "none") for cmd in commands],
                "audio": "data:audio/mp3;base64," + encoded_audio
            })

        await asyncio.sleep(request_data.delay)


# --- 상태 모니터링 ---
async def monitor_connection_status():
    global last_image_received_time

    while True:
        try:
            current_time = time.time()

            # 이미지 수신 상태 확인
            if last_image_received_time > 0 and current_time - last_image_received_time > 10:
                logger.warning("No images received from JetBot for over 10 seconds")
                await broadcast_to_clients({
                    "response": "Warning: No recent images from JetBot. Check JetBot connection."
                })

            await asyncio.sleep(5)  # 5초마다 확인
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    asyncio.ensure_future(connect_to_driver())
    asyncio.ensure_future(monitor_connection_status())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
