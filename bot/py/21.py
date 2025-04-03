from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import httpx
from pathlib import Path
import edge_tts
import os
import websockets

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "granite3.2-vision"
DRIVER_WEBSOCKET_URL = "ws://192.168.137.181:8766"
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "en-US-JennyNeural"
DELAY_SECONDS = 0.1
MAX_SPEED = 0.4

# --- FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "WEBSOCKET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Pydantic Models ---
class OllamaRequest(BaseModel):
    prompt: str = Field(..., description="The user's text prompt.")
    delay: float = Field(DELAY_SECONDS, description="Delay between actions in seconds.")

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index_0325_1.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index_0325_1.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- TTS Function ---
async def generate_tts(text: str) -> str:
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
        return await generate_tts("TTS failed.")

# --- Ollama Interaction ---
async def query_ollama(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    data = {
        "model": MODEL_NAME,
        "prompt": (
            f"{prompt}\n"
            "You are an AI controlling a Professional Driver vehicle in autonomous mode. Your task is to navigate safely based on the provided image. "
            "Suppose you are driving for a driver with poor eyesight and blind. Therefore, refrain from using words such as 'blurred image' or 'lack clarity' in the explanation field."
            "Analyze the image in extreme detail and describe what is directly ahead of the vehicle, including objects, obstacles, pathways, or hazards. "
            "Estimate distances and sizes in centimeters (cm) based on your best judgment, using common objects for reference if possible. "
            "Express the velocity of the vehicle in meters per second (m/s). "
            "Generate actionable commands for the vehicle in JSON format using these commands: 'forward', 'backward', 'left', 'right', 'stop'. "
            "For each command, include 'speed' (0.0 to 0.4) and 'steering' (-1.0 to 1.0, where -1 is full left, 0 is straight, 1 is full right) in 'parameters'. "
            "Add a 'tts' field with natural, descriptive text explaining why the action is taken. "
            "Prioritize safety: if an obstacle is ahead, avoid it and explain the maneuver in the 'tts'. "
            "Adjust speed and steering based on the situation—slow speed with sharp steering for tight turns, fast speed with slight steering for gentle curves. "
            "Do not treat road lines or lane markings as obstacles; interpret them as part of the path to follow. "
            "Provide various responses such as 'move forward slowly,' 'turn left gently,' 'turn right sharply,' or 'U-turn cautiously' depending on the situation. "
            "If an obstacle appears in front of you, drive at 0.5 times your current speed and adjust steering to avoid it. "
            "For each command, include 'speed' (0.0 to 0.4) and 'steering' (-1.0 to 1.0, where -1 is full left, 0 is straight, 1 is full right) in 'parameters'. "
            "Adjust steering in increments of 0.1 for fine control (e.g., -0.9, -0.8, ..., 0.8, 0.9). "
            "Use this JSON format:\n"
            "```json\n"
            "{\n"
            "  \"commands\": [\n"
            "    {\"command\": \"<command_name>\", \"parameters\": {\"speed\": <float>, \"steering\": <float>}, \"tts\": \"<spoken feedback>\"},\n"
            "    ... more commands ...\n"
            "  ],\n"
            "  \"description\": \"<detailed scene description>\"\n"
            "}\n"
            "```\n"
            "Be accurate, creative, and safe. Focus on what’s directly ahead and respond accordingly."
        ),
        "images": [image_data] if image_data else [],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.5, "top_p": 0.95, "num_predict": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            parsed_response = json.loads(result.get("response", "{}"))
            commands = parsed_response.get("commands", [])
            description = parsed_response.get("description", "No description provided.")
            if not commands:
                raise ValueError("No valid commands returned")
            return {"commands": commands, "description": description}
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return {
            "commands": [{"command": "stop", "parameters": {"speed": 0.0, "steering": 0.0}, "tts": "An error occurred."}],
            "description": f"Error: {str(e)}"
        }

# --- WebSocket Connections ---
client_websockets: List[WebSocket] = []
driver_websocket: Optional[WebSocket] = None
current_image_base64: Optional[str] = None
autonomous_task: Optional[asyncio.Task] = None
is_autonomous_running: bool = False

async def connect_to_driver():
    global driver_websocket
    while True:
        try:
            async with websockets.connect(DRIVER_WEBSOCKET_URL) as websocket:
                driver_websocket = websocket
                logger.info("Connected to Driver WebSocket")
                while True:
                    data = await websocket.recv()
                    message = json.loads(data)
                    if "image" in message:
                        global current_image_base64
                        current_image_base64 = message["image"]
                        await broadcast_to_clients({"image": current_image_base64})
                    logger.debug(f"Received from Driver: {data}")
        except Exception as e:
            logger.error(f"Driver connection failed: {e}")
            driver_websocket = None
            await asyncio.sleep(5)

async def broadcast_to_clients(data: Dict[str, Any]):
    disconnected_clients = []
    for ws in client_websockets:
        try:
            await ws.send_text(json.dumps(data))
        except WebSocketDisconnect:
            disconnected_clients.append(ws)
    for ws in disconnected_clients:
        client_websockets.remove(ws)

@app.websocket("/ws/client")
async def client_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client WebSocket connected")
    client_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
            try:
                message = json.loads(data)
                command = message.get("command", "none")
                parameters = message.get("parameters", {})
                await process_command(command, parameters)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
        client_websockets.remove(websocket)

async def process_command(command: str, parameters: Dict[str, Any]):
    global autonomous_task, is_autonomous_running
    prompt = parameters.get("text", f"Execute {command}")

    if command in ["forward", "backward", "left", "right", "stop", "cruise"]:
        if driver_websocket:
            command_message = json.dumps({
                "command": command,
                "parameters": {"speed": min(0.2, MAX_SPEED), "steering": 0.0}
            })
            await driver_websocket.send(command_message)
            tts_text = f"{command.capitalize()} executed."
            encoded_audio = await generate_tts(tts_text)
            await broadcast_to_clients({
                "response": tts_text,
                "driver_command": command,
                "audio": "data:audio/mp3;base64," + encoded_audio
            })
            await asyncio.sleep(1.1)
        else:
            await broadcast_to_clients({"response": "Driver not connected!", "driver_command": "none"})

    elif command == "describe":
        if not current_image_base64:
            await broadcast_to_clients({"response": "No image available!", "driver_command": "none"})
            return
        ollama_response = await query_ollama(prompt, current_image_base64)
        description = ollama_response.get("description", "No description.")
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
        for cmd in commands:
            driver_command = cmd.get("command", "none")
            cmd_params = cmd.get("parameters", {"speed": 0.2, "steering": 0.0})
            cmd_params["speed"] = min(cmd_params.get("speed", 0.2), MAX_SPEED)
            cmd_params["steering"] = max(min(cmd_params.get("steering", 0.0), 1.0), -1.0)
            tts_text = cmd.get("tts", f"Executing {driver_command}.")
            if driver_websocket and driver_command != "none":
                command_message = json.dumps({"command": driver_command, "parameters": cmd_params})
                await driver_websocket.send(command_message)
                await asyncio.sleep(0.05)
                encoded_audio = await generate_tts(tts_text)
                await broadcast_to_clients({
                    "response": tts_text,
                    "driver_command": driver_command,
                    "audio": "data:audio/mp3;base64," + encoded_audio,
                    "description": description
                })
            else:
                await broadcast_to_clients({"response": "Driver not connected!", "driver_command": "none"})
            await asyncio.sleep(DELAY_SECONDS)

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
                await driver_websocket.send(json.dumps({"command": "stop", "parameters": {"speed": 0.0, "steering": 0.0}}))

# --- Autonomous Control Loop ---
async def autonomous_control(request_data: OllamaRequest):
    global is_autonomous_running
    last_image_hash = None
    while is_autonomous_running:
        logger.info("--- Autonomous mode running ---")
        if not current_image_base64:
            logger.warning("No image available from Driver.")
            await broadcast_to_clients({"driver_command": "none"})
            await asyncio.sleep(request_data.delay)
            continue

        current_hash = hash(current_image_base64)
        if current_hash == last_image_hash:
            await asyncio.sleep(request_data.delay)
            continue
        last_image_hash = current_hash

        ollama_response = await query_ollama(request_data.prompt, current_image_base64)
        commands = ollama_response.get("commands", [])

        # 자율주행 중 TTS만 생성, response와 description은 생략
        combined_tts = " ".join(cmd.get("tts", f"Executing {cmd.get('command', 'none')}.") for cmd in commands if cmd.get("command") != "none")[:3]
        if not combined_tts:
            combined_tts = "No valid actions to perform."

        tts_task = asyncio.create_task(generate_tts(combined_tts))

        for cmd in commands:
            driver_command = cmd.get("command", "none")
            cmd_params = cmd.get("parameters", {"speed": 0.2, "steering": 0.0})
            cmd_params["speed"] = min(cmd_params.get("speed", 0.2), MAX_SPEED)
            cmd_params["steering"] = max(min(cmd_params.get("steering", 0.0), 1.0), -1.0)
            if driver_websocket and driver_command != "none":
                command_message = json.dumps({"command": driver_command, "parameters": cmd_params})
                await driver_websocket.send(command_message)
                await asyncio.sleep(0.05)

        encoded_audio = await tts_task
        # 자율주행 중 response와 description 전송 생략
        await broadcast_to_clients({
            "driver_command": [cmd["command"] for cmd in commands],
            "audio": "data:audio/mp3;base64," + encoded_audio
        })

        await asyncio.sleep(request_data.delay)

@app.on_event("startup")
async def startup_event():
    asyncio.ensure_future(connect_to_driver())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
