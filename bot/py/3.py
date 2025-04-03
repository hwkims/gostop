from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
import json
import logging
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any
from pydantic import BaseModel
from pathlib import Path
import httpx
import edge_tts
import io

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"
VOICE = "ko-KR-HyunsuNeural"
TTS_RATE = "+30%"

app = FastAPI()

# CORS 설정
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


# --- Ollama API 호출 (비동기, 스트리밍) ---
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": True,
        "format": "json",
        "options": {"temperature": 0.2, "top_p": 0.8},
    }

    logger.debug(f"Ollama 요청: 프롬프트: {prompt[:50]}..., 이미지: {'있음' if image_data else '없음'}")

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

              async for chunk in response.aiter_text():
                  try:
                      for line in chunk.splitlines():
                          if line.strip():
                              yield json.loads(line)
                  except json.JSONDecodeError as e:
                      logger.error(f"JSON 디코드 오류: {e}, 청크: {chunk}")

    except httpx.RequestError as e:
        logger.error(f"Httpx 요청 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama API 요청 중 오류: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 상태 오류: {e}, 응답: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API 오류: {e}")
    except Exception as e:
        logger.exception("Ollama 스트리밍 처리 중 예기치 않은 오류")
        raise HTTPException(status_code=500, detail=str(e))


# --- TTS 함수 (비동기) -> BytesIO 반환 ---
async def text_to_speech(text: str) -> io.BytesIO:
    """
    edge-tts를 사용하여 텍스트를 음성으로 변환하고, BytesIO 객체에 저장하여 반환합니다.
    """
    communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    audio_stream.seek(0)
    return audio_stream

# --- API 엔드포인트 ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    image_data = request_data.image.split(",")[1] if request_data.image and request_data.image.startswith("data:image") else None
    full_response_text = ""

    try:
        async for response_part in query_ollama_stream(request_data.prompt, image_data):
            if "response" in response_part:
                full_response_text += response_part["response"]

        if not full_response_text.strip():
            return JSONResponse({"response": "", "prompt": request_data.prompt, "tts_data": ""})

        tts_stream = await text_to_speech(full_response_text)
        tts_data = base64.b64encode(tts_stream.read()).decode("utf-8")

        return JSONResponse({
            "response": full_response_text,
            "prompt": request_data.prompt,
            "tts_data": tts_data,
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("API 요청 처리 중 오류")
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {e}")


# --- 정적 파일 제공 ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
