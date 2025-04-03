from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64
import json
import logging
import asyncio
import time
from typing import Optional, List, Dict, Any, AsyncGenerator
from pydantic import BaseModel
from pathlib import Path
import edge_tts
import uuid
import os
import shutil
import httpx


# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"  # Ollama 주소
MODEL_NAME = "gemma3:4b"  # 사용할 모델 이름
VOICE = "ko-KR-HyunsuNeural"  # TTS 음성 설정
TEMP_DIR = "temp_tts"  # 임시 TTS 파일 저장 디렉토리
TTS_RATE = "+30%"  # TTS 재생 속도 (예: "+30%"는 30% 빠르게)


app = FastAPI()

# CORS 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 ---
class OllamaRequest(BaseModel):
    prompt: str  # 사용자 입력 프롬프트
    image: Optional[str] = None  # 이미지는 base64 문자열 (선택 사항)
    context: Optional[List[int]] = None  # 이전 대화 컨텍스트 (선택 사항)

# --- 대화 컨텍스트 관리를 위한 클래스 ---
class ConversationContext:
    def __init__(self):
        self.context: List[int] = []

    def update(self, new_context: List[int]):
        if new_context:  # 빈 리스트가 아닌 경우에만 업데이트
            self.context = new_context

    def get(self) -> List[int]:
        return self.context

# 대화 컨텍스트 인스턴스 생성 (전역)
conversation = ConversationContext()


# 임시 디렉토리 생성 (앱 시작 시)
@app.on_event("startup")
async def startup_event():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

# 임시 디렉토리 삭제 (앱 종료 시)
@app.on_event("shutdown")
async def shutdown_event():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

# --- Ollama API 호출 (비동기, 스트리밍) ---
async def query_ollama_stream(prompt: str, image_data: Optional[str] = None, context: Optional[List[int]] = None) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Ollama API를 비동기적으로 호출하고 스트리밍 방식으로 응답을 받습니다.

    Args:
        prompt: 사용자 프롬프트.
        image_data: (선택 사항) base64로 인코딩된 이미지 데이터.
        context: (선택사항) 이전 대화의 컨텍스트

    Yields:
        각 응답 조각 (JSON).

    Raises:
        HTTPException: API 호출 실패 또는 응답 오류 시 발생.
    """
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_data] if image_data else [],
        "stream": True,  # 스트리밍 활성화
        "format": "json",
        "options": {"temperature": 0.2, "top_p": 0.8},
        "context": context or [], # 컨텍스트 사용
    }

    logger.debug(f"Ollama에 스트리밍 요청 전송: 프롬프트 (처음 50자): {prompt[:50]}, 이미지 (처음 50자): {image_data[:50] if image_data else '이미지 없음'}")

    try:
        async with httpx.AsyncClient() as client:
          async with client.stream(
              "POST",
              f"{OLLAMA_HOST}/api/generate",
              headers={"Content-Type": "application/json"},
              json=data,
              timeout=httpx.Timeout(60.0, connect=5.0, read=60.0, write=5.0),

          ) as response:

              response.raise_for_status()

              async for chunk in response.aiter_bytes():
                  # 각 청크를 JSON으로 파싱 (여러 개의 JSON 객체가 연결된 형태)
                  for part in chunk.decode().strip().split("\n"):
                      if part:  # 빈 문자열은 건너뜀
                          try:
                              yield json.loads(part)
                          except json.JSONDecodeError as e:
                              logger.error(f"JSON 디코드 오류: {e}, 청크 부분: {part}")
                              # 오류 발생 시에도 계속 진행 (필요에 따라 중단 가능)

    except httpx.RequestError as e:
        logger.error(f"Httpx 요청 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Ollama API 요청 중 오류 발생: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 상태 오류: {e}, 응답: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API 오류: {e}")

    except Exception as e:
        logger.exception("Ollama 스트리밍 처리 중 예기치 않은 오류 발생")
        raise HTTPException(status_code=500, detail=f"Ollama 스트리밍 처리 중 오류: {e}")



# --- TTS 함수 (비동기) ---
async def text_to_speech(text: str, output_filename: str) -> None:
    """
    주어진 텍스트를 음성으로 변환하여 파일에 저장합니다. (edge-tts 사용)

    Args:
        text: 음성으로 변환할 텍스트.
        output_filename: 저장할 오디오 파일 경로.
    """
    communicate = edge_tts.Communicate(text, VOICE, rate=TTS_RATE)  # rate 설정
    await communicate.save(output_filename)

# --- API 엔드포인트 (비동기, 스트리밍) ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    """
    이미지 또는 텍스트 프롬프트를 받아 Ollama에 비동기적으로 요청(스트리밍)하고,
    응답을 처리하여 TTS를 생성, 이전 대화내용 context를 고려하여 대화합니다.

    Args:
        request_data: 요청 데이터 (OllamaRequest 모델).

    Returns:
        스트리밍 응답 (각 청크는 JSON 객체).

    Raises:
        HTTPException: API 호출 또는 처리 중 오류 발생 시.
    """
    # data URL 프리픽스 제거 (필요한 경우)
    image_data = request_data.image
    if image_data and image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]


    full_response = []
    async for ollama_response_part in query_ollama_stream(request_data.prompt, image_data, conversation.get()):
        full_response.append(ollama_response_part)  # 응답 조각 저장

        if "done" in ollama_response_part and ollama_response_part["done"]:
            conversation.update(ollama_response_part.get("context"))
            break


    full_response_text = "".join([part.get('response', '') for part in full_response])

    if not full_response_text.strip():
      return {"response": "", "tts_url": "", "prompt": request_data.prompt}

    # TTS 생성 (비동기)
    tts_filename = f"tts_{uuid.uuid4()}.mp3"
    tts_filepath = Path(TEMP_DIR) / tts_filename

    try:
      await text_to_speech(full_response_text, str(tts_filepath))

      return {
        "response": full_response_text,
        "tts_url": f"/tts/{tts_filename}",
        "prompt": request_data.prompt,
      }

    except Exception as e:
        logger.exception("TTS 생성 중 오류 발생")
        raise HTTPException(status_code=500, detail=f"TTS 생성 오류: {e}")


# --- TTS 파일 제공 엔드포인트 ---
@app.get("/tts/{filename}")
async def get_tts(filename: str):
    """
    TTS 파일을 제공합니다.

    Args:
        filename: 요청하는 TTS 파일 이름.

    Returns:
        FileResponse: TTS 파일.

    Raises:
        HTTPException: 파일을 찾을 수 없을 때.
    """
    file_path = Path(TEMP_DIR) / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="TTS 파일을 찾을 수 없습니다.")
    return FileResponse(file_path)


# --- 정적 파일 제공 --- (FastAPI에서 직접 제공, /static 경로)
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    메인 페이지 (index.html)를 반환합니다.
    """
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html 파일을 찾을 수 없습니다.")

# /static 경로로 정적 파일 제공 (HTML에서 CSS, JS 파일 로드)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
