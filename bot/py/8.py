from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator, Dict, Any
from pydantic import BaseModel
import httpx
import websockets
import edge_tts
import base64
import io
from pathlib import Path
import os
import unicodedata

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"
VOICE = "ko-KR-SunHiNeural"
TTS_RATE = "+10%"
TTS_PITCH = "+0%"

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 ---
class OllamaRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    action: str = "navigate"
    direction_hint: Optional[str] = None

# --- Ollama API ---
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None) -> AsyncGenerator[str, None]:
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0.1, "top_p": 0.7},
    }
    logger.debug(f"Ollama 요청: {prompt[:50]}..., 이미지: {'있음' if image_data else '없음'}")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/generate",
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=httpx.Timeout(60.0, connect=5.0, read=120.0, write=5.0),
            ) as response:
                response.raise_for_status()
                full_response = ""
                async for chunk in response.aiter_bytes():  # aiter_bytes() 사용
                    try:
                        decoded_chunk = chunk.decode('utf-8')  # UTF-8로 디코딩
                        for line in decoded_chunk.splitlines():
                            if line.strip():
                                if line.strip().startswith('{') and line.strip().endswith('}'):
                                    json_part = json.loads(line)
                                    if "response" in json_part:
                                        full_response += json_part["response"]

                                    if  "done" in json_part and json_part["done"]:
                                        full_response = unicodedata.normalize('NFC', full_response)  # 여기서 처리
                                        yield full_response
                                        full_response = ""

                                else:
                                    logger.warning(f"Received non-JSON data: {line.strip()}")
                    except UnicodeDecodeError as e:
                        logger.error(f"UTF-8 Decode Error: {e}, chunk: {chunk.decode('latin-1', errors='ignore')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON Decode Error: {e}, chunk: {decoded_chunk}")
                        logger.error(f"Raw chunk causing error: {chunk.decode('utf-8', errors='ignore')}")

    except httpx.RequestError as e:
        logger.error(f"httpx RequestError: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"httpx HTTP Status Error: {e}, Response: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during Ollama streaming")
        raise HTTPException(status_code=500, detail=str(e))

# --- TTS ---
async def speak(text: str) -> str:
    try:
        voice_settings = f'<voice name="{VOICE}"><prosody rate="{TTS_RATE}" pitch="{TTS_PITCH}">{text}</prosody></voice>'
        communicate = edge_tts.Communicate(text=voice_settings, voice=VOICE)

        temp_file = "temp_audio.wav"
        await communicate.save(temp_file)
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        os.remove(temp_file)
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")

# --- JetBot 명령 ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None):
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
            await websocket.send(json.dumps({"command": command, "parameters": parameters}))
            logger.info(f"Sent command to JetBot: {command}, {parameters}")
    except Exception as e:
        logger.error(f"Failed to send command to JetBot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FastAPI 엔드포인트 ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_data = request_data.image.split(",")[1] if request_data.image and request_data.image.startswith("data:image") else None
    full_response_text = ""  # 최종 응답 저장

    try:
        async for response_str in query_ollama_stream(request_data.prompt, image_data):
            full_response_text = response_str  # 스트림에서 받은 최종 응답 저장
            logger.info(f"응답: {full_response_text}")

        if not full_response_text:
            return JSONResponse({"response": "Ollama로부터 받은 정보가 없습니다.", "jetbot_command": "none"})


        # JetBot 명령 결정 로직
        jetbot_command = "none"
        parameters = {}

        if request_data.action == "navigate":
            if request_data.direction_hint == "left":
                jetbot_command = "turn_left"
            elif request_data.direction_hint == "right":
                jetbot_command = "turn_right"
            elif request_data.direction_hint == "forward":
                jetbot_command = "move_forward"
            elif request_data.direction_hint == "backward":
                jetbot_command = "move_backward"
            elif request_data.direction_hint == "stop":
                jetbot_command = "stop"
            elif "obstacle" in full_response_text.lower() or "object" in full_response_text.lower(): # 수정
                jetbot_command = "avoid_obstacle"
                parameters = {"direction": "left"}

            # 추가 명령 (정지, 회전, 천천히)
            elif request_data.direction_hint == "rotate_clockwise":
                jetbot_command = "rotate"
                parameters = {"angle": 90}
            elif request_data.direction_hint == "rotate_counterclockwise":
                jetbot_command = "rotate"
                parameters = {"angle": -90}
            elif request_data.direction_hint == "forward_slow":
                jetbot_command = "move_forward"
                parameters = {"speed": 0.3}
            elif request_data.direction_hint == "backward_slow":
                jetbot_command = "move_backward"
                parameters = {"speed": 0.3}
            elif request_data.direction_hint == "random":
                jetbot_command = "random_action"

        elif request_data.action == "describe":
            jetbot_command = "none"
        elif request_data.action == "custom":
            jetbot_command = "custom_command"
            parameters = {"prompt": request_data.prompt}

        await send_command_to_jetbot(jetbot_command, parameters)

        # TTS
        encoded_audio = await speak(full_response_text)  # 최종 응답 사용

        return JSONResponse({
            "response": full_response_text,  # 최종 응답
            "jetbot_command": jetbot_command,
            "audio": encoded_audio
        })

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"Error during API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 정적 파일 ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
