# main.py (PC - FastAPI + Ollama + JetBot Command + Static Files + TTS)

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
import re

# --- 로깅 ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
JETBOT_WEBSOCKET_URL = "ws://192.168.137.233:8766"  # JetBot IP 주소
VOICE = "ko-KR-HyunsuNeural"  # 더 자연스러운 한국어 음성
# TTS_RATE = "+10%"  # 제거
# TTS_PITCH = "+0%" # 제거

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 ---
class OllamaRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    action: str = "navigate"  # 기본 액션
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
                async for chunk in response.aiter_bytes():
                    try:
                        decoded_chunk = chunk.decode('utf-8')
                        for line in decoded_chunk.splitlines():
                            if line.strip():
                                if line.strip().startswith('{') and line.strip().endswith('}'):
                                    json_part = json.loads(line)
                                    if "response" in json_part:
                                        full_response += json_part["response"]
                                    if json_part.get("done"):
                                        full_response = unicodedata.normalize('NFC', full_response)
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
        # voice_settings = f'<voice name="{VOICE}"><prosody rate="{TTS_RATE}" pitch="{TTS_PITCH}">{text}</prosody></voice>' #제거
        # communicate = edge_tts.Communicate(text=voice_settings, voice=VOICE)
        communicate = edge_tts.Communicate(text=text, voice=VOICE) # 수정

        temp_file = "temp_audio.wav"
        await communicate.save(temp_file)
        with open(temp_file, "rb") as f:
            audio_data = f.read()
        os.remove(temp_file)
        return base64.b64encode(audio_data).decode("utf-8")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS Error: {e}")

# --- JetBot 명령 전송 ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None, timeout: float = 10.0):
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL, open_timeout=timeout) as websocket:
            await websocket.send(json.dumps({"command": command, "parameters": parameters}))
            logger.info(f"Sent command to JetBot: {command}, {parameters}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"JetBot connection closed unexpectedly: {e}")
        raise HTTPException(status_code=500, detail="JetBot connection error")
    except asyncio.TimeoutError:
        logger.error(f"Timeout connecting to JetBot at {JETBOT_WEBSOCKET_URL}")
        raise HTTPException(status_code=504, detail="JetBot connection timeout")
    except Exception as e:
        logger.error(f"Failed to send command to JetBot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FastAPI 엔드포인트 ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_data = request_data.image.split(",")[1] if request_data.image and request_data.image.startswith("data:image") else None
    full_response_text = ""

    try:
        async for response_str in query_ollama_stream(request_data.prompt, image_data):
            full_response_text = response_str
            logger.info(f"응답: {full_response_text}")

        if not full_response_text:
            return JSONResponse({"response": "Ollama로부터 받은 정보가 없습니다.", "jetbot_command": "none"})


        # JetBot 명령 결정 로직
        jetbot_command = "none"
        parameters = {}
        tts_text = ""

        if request_data.action == "navigate":
            if request_data.direction_hint == "left":
                jetbot_command = "left_medium"
                tts_text = "왼쪽으로 회전합니다."
            elif request_data.direction_hint == "right":
                jetbot_command = "right_medium"
                tts_text = "오른쪽으로 회전합니다."
            elif request_data.direction_hint == "forward":
                jetbot_command = "forward_medium"
                tts_text = "앞으로 이동합니다."
            elif request_data.direction_hint == "backward":
                jetbot_command = "backward_medium"
                tts_text = "뒤로 이동합니다."
            elif request_data.direction_hint == "stop":
                jetbot_command = "stop"
                tts_text = "정지합니다."
            elif "obstacle" in full_response_text.lower() or "object" in full_response_text.lower():
                jetbot_command = "avoid_obstacle"
                parameters = {"direction": "left"}
                tts_text = "장애물을 피해 왼쪽으로 이동합니다."

            # 추가 명령 (회전, 천천히, 랜덤)
            elif request_data.direction_hint == "rotate_clockwise":
                jetbot_command = "right_fast"
                parameters = {"duration": 1.4}
                tts_text = "시계 방향으로 회전합니다."
            elif request_data.direction_hint == "rotate_counterclockwise":
                jetbot_command = "left_fast"
                parameters = {"duration": 1.4}
                tts_text = "반시계 방향으로 회전합니다."
            elif request_data.direction_hint == "forward_slow":
                jetbot_command = "forward_slow"
                tts_text = "천천히 앞으로 이동합니다."
            elif request_data.direction_hint == "backward_slow":
                jetbot_command = "backward_slow"
                tts_text = "천천히 뒤로 이동합니다."
            elif request_data.direction_hint == "random":
                jetbot_command = "random_action"
                tts_text = "랜덤 동작을 수행합니다."

        elif request_data.action == "describe":
            jetbot_command = "none"
            tts_text = full_response_text  # 전체 응답

        elif request_data.action == "custom":
            jetbot_command = "custom_command"
            parameters = {"prompt": request_data.prompt}
            tts_text = f"사용자 지정 명령: {request_data.prompt}"

        await send_command_to_jetbot(jetbot_command, parameters)

        # TTS 출력 생성
        encoded_audio = await speak(tts_text)

        return JSONResponse({
            "response": full_response_text,  # 전체
            "jetbot_command": jetbot_command, # Jetbot 커맨드
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
