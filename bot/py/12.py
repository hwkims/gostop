from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import os
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import httpx
import websockets
from pathlib import Path
import edge_tts

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 환경 설정 ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:4b"  # 또는 다른 적절한 모델
JETBOT_WEBSOCKET_URL = "ws://192.168.137.181:8766"  # JetBot WebSocket 주소
STATIC_DIR = Path(__file__).parent / "static"
MEMORY_FILE = "memory.json"
TTS_VOICE = "ko-KR-HyunsuNeural"  # 또는 다른 TTS 음성

# --- FastAPI 설정 ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# --- 데이터 모델 ---
class OllamaRequest(BaseModel):
    prompt: str  # 사용자 프롬프트
    image: Optional[str] = None  # 이미지 데이터 (base64) - 사용되지 않음. Jetbot에서 직접 받음
    action: str = "navigate"  # 액션 (navigate, describe)
    direction_hint: Optional[str] = None  # 방향 힌트 (left_medium, right_medium, ...)
    speed: Optional[float] = None  # 속도 (옵션)
    duration: Optional[float] = None  # 지속 시간 (옵션)
    angle: Optional[float] = None # 각도

# --- JetBot 명령어 ---
JETBOT_COMMANDS = {
    "left_medium": {"command": "left", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "왼쪽으로 살짝 이동!"},
    "right_medium": {"command": "right", "parameters": {"speed": 0.3, "duration": 0.7}, "tts": "오른쪽으로 살짝 이동!"},
    "forward_medium": {"command": "forward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "앞으로 이동!"},
    "backward_medium": {"command": "backward", "parameters": {"speed": 0.4, "duration": 1.0}, "tts": "뒤로 이동!"},
    "stop": {"command": "stop", "parameters": {}, "tts": "정지!"},
    "dance": {"command": "dance", "parameters": {}, "tts": "춤을 춥니다!"},
    "none": {"command": "none", "parameters": {}, "tts": "대기 중."},  # 'none' 추가
}

# --- TTS 함수 ---
async def generate_tts(text: str) -> str:
    """주어진 텍스트를 사용하여 TTS 음성을 생성하고, base64로 인코딩된 오디오 데이터를 반환합니다."""
    try:
        if not text or text.isspace():
            text = "처리할 내용이 없습니다."  # 빈 텍스트 처리
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        temp_file = "temp_tts_" + str(int(time.time())) + ".mp3"
        temp_filepath = STATIC_DIR / temp_file
        await communicate.save(temp_filepath)

        with open(temp_filepath, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        os.remove(temp_filepath)  # 임시 파일 삭제
        logger.info(f"TTS successfully generated for: {text}")
        return audio_data
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return await generate_tts("음성 생성 중 오류가 발생했습니다.")

# --- Gemma 3 상호작용 함수 ---
async def query_gemma3(prompt: str, image_data: Optional[str] = None, action: Optional[str] = None) -> Dict[str, Any]:
    """
    Gemma 3 모델에 쿼리를 보내고 응답을 반환합니다.

    Args:
        prompt: 사용자 프롬프트.
        image_data: base64로 인코딩된 이미지 데이터 (선택 사항).
        action: 수행할 액션 ('describe' 또는 'navigate').

    Returns:
        Gemma 3 모델의 응답 (JSON 형식).
    """
    if action == 'describe':
        # 이미지 설명 프롬프트 (객체 중심, 예시 없음, 네비게이션 없음)
        base_prompt = (
            "다음은 base64로 인코딩된 이미지입니다. 이미지의 내용을 자세히 설명해주세요.\n"
            "반드시 다음 사항을 포함해야 합니다:\n"
            "- 각 객체의 종류, 색상, 모양, 상대적 위치.\n"
            "- 이미지에 텍스트가 있다면, 텍스트 내용, 글꼴 스타일, 색상, 위치.\n"
            "객관적이고 상세하게 묘사하고, 분위기나 느낌은 설명하지 마세요. "
            "'forward', 'backward', 'left', 'right', 'stop', 'dance'와 같은 명령을 제안하지 마세요.\n"
            "응답은 다음 JSON 형식을 엄격히 준수해야 합니다:\n"
            "{'commands': [{'command': 'describe', 'parameters': {}, 'tts': '이미지 설명'}]}"
        )
        if image_data:
            base_prompt = base_prompt.replace("다음은 base64로 인코딩된 이미지입니다.", f"다음은 base64로 인코딩된 이미지입니다: {image_data}")

    elif image_data:
        # 이미지 기반 프롬프트 + 사용자 프롬프트 (네비게이션 및 기타 액션)
        base_prompt = (
            f"다음은 base64로 인코딩된 이미지입니다: {image_data}\n"
            f"사용자 요청: '{prompt}'.\n"
            "이미지와 사용자 요청을 기반으로 JetBot이 수행할 적절한 행동 '하나'를 제안하세요.\n"
            "- 이미지 내 객체 정보(종류, 색상, 모양, 위치), 배경, 텍스트(있는 경우)를 고려하세요.\n"
            "명령은 'forward', 'backward', 'left', 'right', 'stop', 'dance', 'describe' 중 하나여야 합니다.\n"
            "'forward', 'backward', 'left', 'right' 명령은 'speed'(0.3~0.7), 'duration'(0.5~3.0)을 포함해야 합니다.\n"
            "tts 필드에는 JetBot의 행동 설명을 넣어주세요.\n"
            "응답 JSON 형식: {'commands': [{'command': '명령', 'parameters': {'speed': 값, 'duration': 값}, 'tts': '설명'}]}"
        )
    else:
        # 이미지가 없는 경우 (텍스트 전용 명령)
        base_prompt = (
            f"사용자 요청: '{prompt}'.\n"
            "JetBot이 수행할 적절한 행동 '하나'를 JSON 형식으로 제안하세요.\n"
            "응답 JSON 형식: {'commands': [{'command': '명령', 'parameters': {'speed': 값, 'duration': 값}, 'tts': '설명'}]}\n"
            "명령은 'forward', 'backward', 'left', 'right', 'stop', 'dance', 'describe' 중 하나.\n"
            "'forward', 'backward', 'left', 'right'는 'speed'(0.3~0.7), 'duration'(0.5~3.0)을 포함.\n"
            "tts는 간결하게 JetBot 행동을 설명해야 합니다."
        )

    data = {
        "model": MODEL_NAME,
        "prompt": base_prompt,
        "images": [],  # 이미지는 prompt에 직접 포함됨
        "stream": False,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(OLLAMA_HOST + "/api/generate", json=data)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            result = response.json()
            # 응답 파싱 및 기본값 설정
            parsed_response = json.loads(result.get("response", "{}")).get("commands", [])
            logger.info(f"Gemma3 response: {parsed_response}")
            return {"commands": parsed_response}

    except httpx.HTTPError as e:
        logger.error(f"Gemma3 HTTP error: {e}, Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "Gemma3 통신 오류! 잠시 멈춤."}]}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Gemma3 response parsing error: {e}, Raw response: {result.get('response', 'No response')}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "응답 처리 오류! 잠시 멈춤."}]}
    except Exception as e:
        logger.error(f"Gemma3 error: {e}")
        return {"commands": [{"command": "stop", "parameters": {}, "tts": "Gemma3 오류 발생! 잠시 멈춤."}]}

# --- JetBot 통신 함수 ---
async def send_command_to_jetbot(command: str, parameters: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """JetBot에 명령을 보내고, 응답으로 받은 이미지를 base64로 인코딩하여 반환합니다."""
    try:
        async with websockets.connect(JETBOT_WEBSOCKET_URL) as websocket:
            msg = {"command": command, "parameters": parameters or {}, "get_image": True}
            await websocket.send(json.dumps(msg))
            logger.info(f"Sent command to JetBot: {msg}")
            response = await websocket.recv()
            data = json.loads(response)
            image = data.get("image")

            if image:
                return image
            else:
                logger.warning("No image received from JetBot")
                return None  # 이미지가 없는 경우

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"JetBot connection closed unexpectedly: {e}")
        return None  # 연결 오류
    except Exception as e:
        logger.error(f"Error communicating with JetBot: {e}")
        return None  # 기타 오류

# --- 메모리 함수 ---
def load_memory(filename: str = MEMORY_FILE) -> List[Dict[str, Any]]:
    """메모리 파일에서 이전 대화 기록을 로드합니다."""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)[-50:]  # 최대 50개 항목 유지
        return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading memory: {e}")
        return []

def save_memory(memory_entry: Dict[str, Any], filename: str = MEMORY_FILE):
    """메모리 파일에 대화 기록을 저장합니다."""
    memory = load_memory(filename)
    memory.append(memory_entry)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
    except OSError as e:
        logger.error(f"Error saving memory: {e}")

# --- API 엔드포인트 (/api/generate) ---
@app.post("/api/generate")
async def generate(request_data: OllamaRequest):
    """
    사용자 요청을 처리하고, Gemma 3 모델과 JetBot을 제어하여 응답을 생성합니다.
    """
    # 1. JetBot으로부터 초기 이미지 가져오기 (항상 'none' 명령 실행)
    image_base64 = await send_command_to_jetbot("none", {})

    # 이미지 데이터 로깅 (처음 100자만)
    if image_base64:
        logger.info(f"Initial image data from JetBot (first 100 chars): {image_base64[:100]}...")
    else:
        logger.warning("No initial image received from JetBot.")

    # 2. 주요 로직 처리
    if request_data.action == 'describe':
        # Describe 액션: 이미지 설명만 생성 (사용자 프롬프트 무시)
        gemma_response = await query_gemma3("", image_base64, action='describe')
        if gemma_response and gemma_response.get("commands"):
             cmd = gemma_response["commands"][0]
             jetbot_command = cmd["command"]
             parameters = cmd["parameters"]
             tts_text = cmd["tts"]
        else:
            jetbot_command = "none"
            parameters = {}
            tts_text = "이미지 설명 생성 실패"


    elif request_data.action == "navigate" and request_data.direction_hint in JETBOT_COMMANDS:
        # Navigate 액션 + 방향 힌트: 힌트 사용, Gemma 3가 재정의 가능
        cmd_info = JETBOT_COMMANDS[request_data.direction_hint]
        default_command = cmd_info["command"]
        default_parameters = cmd_info["parameters"].copy()
        default_tts = cmd_info["tts"]

        # Gemma 3에 이미지와 사용자 프롬프트(버튼에서 옴)를 함께 쿼리
        gemma_response = await query_gemma3(request_data.prompt, image_base64, action="navigate")

        # Gemma 3가 유효한 명령을 제공하면 사용, 그렇지 않으면 기본값 사용
        if gemma_response and gemma_response.get("commands") and len(gemma_response.get("commands"))>0:
            cmd = gemma_response["commands"][0]
            jetbot_command = cmd["command"]
            parameters = cmd["parameters"]
            tts_text = cmd["tts"]
        else:
            jetbot_command = default_command
            parameters = default_parameters
            tts_text = default_tts

        # 요청 데이터에서 speed, duration, angle 값으로 파라미터 재정의
        if request_data.speed is not None:
            parameters["speed"] = request_data.speed
        if request_data.duration is not None:
            parameters["duration"] = request_data.duration
        if request_data.angle is not None:
            parameters["angle"] = request_data.angle

    else:
        # 커스텀 명령 또는 기타 액션: 요청의 프롬프트 사용
        gemma_response = await query_gemma3(request_data.prompt, image_base64)
        commands = gemma_response.get("commands", []) or [{"command": "none", "parameters": {}, "tts": "명령 대기 중"}]
        cmd = commands[0]
        jetbot_command = cmd["command"]
        parameters = cmd["parameters"]
        tts_text = cmd["tts"]

    # 3. JetBot에 명령 실행, 새 이미지 받기
    new_image_base64 = await send_command_to_jetbot(jetbot_command, parameters)
    if not new_image_base64:
        logger.warning("No image received after executing command. Using previous image.")

    # 4. TTS 생성
    encoded_audio = await generate_tts(tts_text)

    # 5. 응답 준비
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    response = {
        "response": tts_text,
        "jetbot_command": jetbot_command,
        "image": "data:image/jpeg;base64," + (new_image_base64 or image_base64),  # 새 이미지가 없으면 이전 이미지 사용
        "audio": "data:audio/mp3;base64," + encoded_audio,
    }
    # 6. 메모리에 저장
    save_memory({
        "timestamp": time.time(),
        "prompt": request_data.prompt,
        "action": request_data.action,
        "direction_hint": request_data.direction_hint,
        "jetbot_command": jetbot_command,
        "tts_text": tts_text,
    })

    return JSONResponse(content=response, headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
