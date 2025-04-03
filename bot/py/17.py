from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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
JETBOT_WEBSOCKET_URL = "ws://192.168.137.181:8766"
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "en-US-JennyNeural"
DEFAULT_ITERATIONS = 1
DELAY_SECONDS = 0.1
AUTONOMOUS_INTERVAL = 5  # 자율 주행 이미지 처리 간격(초)

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
    iterations: int = Field(DEFAULT_ITERATIONS, description="Number of iterations.")
    delay: float = Field(DELAY_SECONDS, description="Delay between actions in seconds.")


# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
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


# --- Ollama Interaction (Autonomous Only) ---
async def query_ollama(prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
    data = {
        "model": MODEL_NAME,
        "prompt": (
            f"{prompt}\n"
            "You are an AI controlling a JetBot robot in autonomous mode. Your task is to navigate safely based on the provided image. "
            "Analyze the image in extreme detail and describe what is directly ahead of the robot, including objects, obstacles, pathways, or hazards. "
            "Estimate distances and sizes in centimeters (cm) based on your best judgment, using common objects for reference if possible. "
            "Generate actionable commands for the JetBot in JSON format using these commands: 'forward', 'backward', 'left', 'right', 'stop', or 'dance'. "
            "For each command, include 'speed' (0.0 to 0.4) and 'duration' (0.0 to 0.4) in 'parameters'. "
            "Add a 'tts' field with natural, descriptive text explaining why the action is taken. "
            "Prioritize safety: if an obstacle is ahead, avoid it and explain the maneuver in the 'tts'. "
            "Adjust speed and duration based on the situation—slow and short for tight spaces, fast and long for open areas. "
            "Use this JSON format:\n"
            "```json\n"
            "{\n"
            "  \"commands\": [\n"
            "    {\"command\": \"<command_name>\", \"parameters\": {\"speed\": <float>, \"duration\": <float>}, \"tts\": \"<spoken feedback>\"},\n"
            "    ... more commands ...\n"
            "  ],\n"
            "  \"description\": \"<detailed scene description>\"\n"
            "}\n"
            "```\n"
            "Be accurate, creative, and safe. Focus on what's directly ahead and respond accordingly."
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
            "commands": [{"command": "stop", "parameters": {}, "tts": "An error occurred."}],
            "description": f"Error: {str(e)}"
        }


# --- WebSocket Connections ---
client_websockets: List[WebSocket] = []
jetbot_websocket: Optional[WebSocket] = None
current_image_base64: Optional[str] = None
autonomous_mode_active = False
autonomous_task = None
last_processed_image = None  # 마지막으로 처리된 이미지 저장


async def connect_to_jetbot():
    global jetbot_websocket
    while True:
        try:
            async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
                jetbot_websocket = websocket
                logger.info("Connected to Jetbot WebSocket")
                while True:
                    data = await websocket.recv()
                    message = json.loads(data)
                    if "image" in message:
                        global current_image_base64
                        current_image_base64 = message["image"]
                        await broadcast_to_clients({"image": current_image_base64})
                    logger.debug(f"Received from Jetbot: {data}")
        except Exception as e:
            logger.error(f"Jetbot connection failed: {e}")
            jetbot_websocket = None
            await asyncio.sleep(5)


# 5초마다 자율주행 처리를 위한 함수
async def autonomous_loop():
    """5초마다 이미지를 캡처하고 처리하는 루프"""
    global autonomous_mode_active, last_processed_image

    logger.info("Starting autonomous loop")

    while autonomous_mode_active:
        if current_image_base64:
            # 현재 이미지 캡처 및 저장
            last_processed_image = current_image_base64

            # 이미지 처리 및 명령 실행
            await process_captured_image(last_processed_image)

            # 다음 처리까지 대기
            logger.info(f"Waiting {AUTONOMOUS_INTERVAL} seconds until next image processing")
            await broadcast_to_clients({
                "response": f"다음 이미지 처리까지 {AUTONOMOUS_INTERVAL}초 대기 중...",
                "processing_status": "waiting"
            })

        # 5초 대기
        await asyncio.sleep(AUTONOMOUS_INTERVAL)

    logger.info("Autonomous loop stopped")


async def process_captured_image(image_data):
    """캡처된 이미지를 처리하고 명령 실행"""
    if not autonomous_mode_active or not image_data:
        return

    try:
        # 이미지 처리 로그 기록
        current_time = time.time()
        logger.info(f"Processing captured image at {current_time}")

        # 클라이언트에 처리 상태 알림
        await broadcast_to_clients({
            "response": "이미지 분석 중...",
            "processing_status": "analyzing"
        })

        # 빠른 처리를 위한 짧은 프롬프트 사용
        prompt = "Navigate safely in autonomous mode"
        # Ollama에 이미지 전송하여 분석 요청
        ollama_response = await query_ollama(prompt, image_data)
        # 응답에서 명령과 설명 추출
        commands = ollama_response.get("commands", [])
        description = ollama_response.get("description", "No description.")

        if commands:
            # 응답성 유지를 위해 첫 번째 명령만 실행
            cmd = commands[0]
            jetbot_command = cmd.get("command", "none")
            cmd_params = cmd.get("parameters", {})
            tts_text = cmd.get("tts", f"Executing {jetbot_command}.")

            # JetBot에 명령 전송
            if jetbot_websocket and jetbot_command != "none":
                command_message = json.dumps({"command": jetbot_command, "parameters": cmd_params})
                await jetbot_websocket.send(command_message)

                # TTS 생성
                encoded_audio = await generate_tts(tts_text)
                # 클라이언트에 결과 전송
                await broadcast_to_clients({
                    "response": tts_text,
                    "jetbot_command": jetbot_command,
                    "audio": "data:audio/mp3;base64," + encoded_audio,
                    "description": description,
                    "processing_status": "completed",
                    "processed_at": time.strftime("%H:%M:%S", time.localtime(current_time))
                })

                # 명령 실행 시간 대기
                await asyncio.sleep(cmd_params.get("duration", 0.4) + 0.1)
    except Exception as e:
        # 오류 발생 시 로그 기록
        logger.error(f"Image processing error: {e}")
        await broadcast_to_clients({
            "response": f"이미지 처리 오류: {str(e)}",
            "processing_status": "error"
        })


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
                iterations = parameters.get("iterations", DEFAULT_ITERATIONS)
                await process_command(command, parameters, iterations)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
        client_websockets.remove(websocket)


async def process_command(command: str, parameters: Dict[str, Any], iterations: int):
    global autonomous_mode_active, autonomous_task

    prompt = parameters.get("text", f"Execute {command}")

    # Direct button commands (no vision)
    if command in ["forward", "backward", "left", "right", "stop", "dance"]:
        # Stop autonomous mode if it's running
        if autonomous_mode_active:
            autonomous_mode_active = False
            if autonomous_task and not autonomous_task.done():
                autonomous_task.cancel()
            await broadcast_to_clients({"response": "자율 주행 모드가 중지되었습니다", "jetbot_command": "none"})

        for _ in range(iterations):
            if jetbot_websocket:
                command_message = json.dumps({
                    "command": command,
                    "parameters": {"speed": 0.35, "duration": 0.4}  # Default parameters
                })
                await jetbot_websocket.send(command_message)
                tts_text = f"{command.capitalize()} executed."
                encoded_audio = await generate_tts(tts_text)
                await broadcast_to_clients({
                    "response": tts_text,
                    "jetbot_command": command,
                    "audio": "data:audio/mp3;base64," + encoded_audio
                })
                await asyncio.sleep(1.1)  # Duration + buffer
            else:
                await broadcast_to_clients({"response": "JetBot not connected!", "jetbot_command": "none"})
            await asyncio.sleep(DELAY_SECONDS)

    # Describe with vision
    elif command == "describe":
        # Stop autonomous mode if it's running
        if autonomous_mode_active:
            autonomous_mode_active = False
            if autonomous_task and not autonomous_task.done():
                autonomous_task.cancel()
            await broadcast_to_clients({"response": "자율 주행 모드가 중지되었습니다", "jetbot_command": "none"})

        if not current_image_base64:
            await broadcast_to_clients({"response": "No image available!", "jetbot_command": "none"})
            return
        ollama_response = await query_ollama(prompt, current_image_base64)
        description = ollama_response.get("description", "No description.")
        encoded_audio = await generate_tts(description)
        await broadcast_to_clients({
            "response": description,
            "jetbot_command": "none",
            "audio": "data:audio/mp3;base64," + encoded_audio,
            "description": description
        })

    # Custom command with vision
    elif command == "custom":
        # Stop autonomous mode if it's running
        if autonomous_mode_active:
            autonomous_mode_active = False
            if autonomous_task and not autonomous_task.done():
                autonomous_task.cancel()
            await broadcast_to_clients({"response": "자율 주행 모드가 중지되었습니다", "jetbot_command": "none"})

        if not current_image_base64:
            await broadcast_to_clients({"response": "No image available!", "jetbot_command": "none"})
            return
        for _ in range(iterations):
            ollama_response = await query_ollama(prompt, current_image_base64)
            commands = ollama_response.get("commands", [])
            description = ollama_response.get("description", "No description.")
            for cmd in commands:
                jetbot_command = cmd.get("command", "none")
                cmd_params = cmd.get("parameters", {})
                tts_text = cmd.get("tts", f"Executing {jetbot_command}.")
                if jetbot_websocket and jetbot_command != "none":
                    command_message = json.dumps({"command": jetbot_command, "parameters": cmd_params})
                    await jetbot_websocket.send(command_message)
                    await asyncio.sleep(cmd_params.get("duration", 1.0) + 0.1)
                    encoded_audio = await generate_tts(tts_text)
                    await broadcast_to_clients({
                        "response": tts_text,
                        "jetbot_command": jetbot_command,
                        "audio": "data:audio/mp3;base64," + encoded_audio,
                        "description": description
                    })
                else:
                    await broadcast_to_clients({"response": "JetBot not connected!", "jetbot_command": "none"})
            await asyncio.sleep(DELAY_SECONDS)

    # Autonomous with vision
    elif command == "autonomous":
        # 이미 자율 주행 모드가 활성화되어 있으면 중지
        if autonomous_mode_active:
            autonomous_mode_active = False
            if autonomous_task and not autonomous_task.done():
                autonomous_task.cancel()
            await broadcast_to_clients({
                "response": "자율 주행 모드가 중지되었습니다",
                "jetbot_command": "none"
            })
            return

        # 자율 주행 모드 시작
        autonomous_mode_active = True
        await broadcast_to_clients({
            "response": "5초마다 이미지를 캡처하여 자율 주행을 시작합니다",
            "jetbot_command": "autonomous"
        })

        # 초기 TTS 안내
        encoded_audio = await generate_tts("5초마다 이미지를 캡처하여 자율 주행을 시작합니다")
        await broadcast_to_clients({
            "audio": "data:audio/mp3;base64," + encoded_audio
        })

        # 자율 주행 루프 시작
        autonomous_task = asyncio.create_task(autonomous_loop())


@app.on_event("startup")
async def startup_event():
    asyncio.ensure_future(connect_to_jetbot())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
