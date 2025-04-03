from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64
import json
import logging
import asyncio
from typing import Optional, Generator, List, Dict, Any
from pydantic import BaseModel
from edge_tts import Communicate
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- 설정 ---
OLLAMA_HOST = "http://localhost:11434"  # Ollama 주소
VOICE = "ko-KR-HyunsuNeural"  # TTS 음성
MODEL_NAME = "gemma3:4b"  # 사용할 모델 (또는 llava 등 멀티모달 모델)

app = FastAPI()

# CORS 설정 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 시스템 프롬프트 (간결하게) ---
SYSTEM_PROMPT = """
(이전의 간결한 프롬프트)
"""
SYSTEM_PROMPT = """
너는 시각 장애인을 위한 안내견 로봇, 범블비야. 사용자를 안전하게 안내하고, 질문에 친절하고 정확하게 답변하며, 쇼핑을 도와줘.

이미지와 사용자 텍스트 입력을 받으면 다음을 수행:

1.  **이미지 분석:** 이미지 속 물체, 위치, 문/손잡이/경첩, 상품 정보, 장면, 로봇 상황 파악.
2.  **명령/질문 해석:** 사용자 의도와 필요 정보 파악.
3.  **행동 계획 (Chain-of-Thought):** 단계별 행동 계획 (동작, 센서, 결과, 고려사항). '안 된다'는 말 대신, 주어진 능력(바퀴) 활용. 조건/반복/지연 동작 가능.
4.  **JSON 출력:** ROS 2 토픽 메시지 (geometry_msgs/Twist 형식 호환).
5.   **JSON 메모리 업데이트:** 현재 상황, 사용자 명령/질문, 수행한 작업, 결과, 생성한 JSON 출력, `speech_output`을 JSON 파일에 새로운 이벤트로 추가해. (구현 필요)

JSON 출력 키:

*   `scene_description`: (문자열) 장면 설명.
*   `objects_detected`: (문자열 리스트) 감지된 물체.
*    `object_locations`: (객체 리스트) 객체 위치
    *   `object`:(문자열) 객체의 이름
    *   `relative_position`: (문자열) 객체의 상대적 위치
     *   `clock_direction`: (문자열) 12시 방향 기준 객체 방향
*   `object_details`: (객체 리스트, 선택) 객체 추가 정보.
    *   `object`: (문자열) 객체 이름.
    *   `details`: (딕셔너리) 객체 추가 정보 (color, size, material, status, price, features).
*   `command_interpretation`: (문자열, 선택) 명령/질문 해석.
*   `action_plan`: (문자열 리스트) 단계별 행동.
*  `plan_explanation`: (문자열) : 행동계획 부가설명.
*   `action_steps`: (객체 리스트) 단계 정의.
    *   `step_id`: (정수) 단계 번호.
    *   `description`: (문자열) 단계 설명.
    *   `type`: (문자열) 단계 유형 (move, turn, wait, check_sensor, speak, other).
    *   `duration`: (실수, 선택) 지속 시간.
    *   `condition`: (문자열, 선택) 조건.
    *   `sensor_type`: (문자열, check_sensor) 센서 종류.
    *   `expected_value`: (문자열, check_sensor) 기대 값.
    *   `motor_commands`: (딕셔너리, 선택) ROS 2 Twist 메시지.
        *   `linear`: (딕셔너리) x, y, z (m/s).
        *   `angular`: (딕셔너리) x, y, z (rad/s).
    *   `repeat`: (정수, 선택) 반복 횟수.
    *  `distance` : (선택사항, 실수) 이동거리
    *  `angle`: (선택사항, 실수) 회전 각도
*   `is_safe`: (boolean) 안전 여부.
*   `speech_output`: (문자열) 로봇 발화.

JSON 메모리 활용, 간결/구체적 설명, 재치/친절 말투, 유머/감정 표현 사용.
"""
# --- Pydantic 모델 ---
class OllamaRequest(BaseModel):
    prompt: str
    image: Optional[str] = None
    context: Optional[List[int]] = None

# --- Ollama API 호출 (Streamlit 방식) ---
async def query_ollama(prompt: str, image_data: Optional[str] = None, context: Optional[List[int]] = None) -> Dict[str, Any]:
    """Ollama API 호출 (Streamlit 방식, 타임아웃, 재시도)"""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "context": context or [],
        "options": {"temperature": 0.2, "top_p": 0.8},
    }

    # Streamlit 코드처럼, base64 문자열을 바로 "images" 리스트에 넣음
    if image_data:
        data["images"] = [image_data]
        logger.debug(f"Sending image to Ollama (first 50 chars): {image_data[:50]}...")

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=data,
                timeout=60,  # 타임아웃 유지
            )
            response.raise_for_status()
            response_json = response.json()
            logger.debug(f"Ollama response: {response_json}")
            return response_json

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e)) from e
            await asyncio.sleep(retry_delay)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response.text}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from Ollama: {e}") from e

# --- TTS 함수 ---
async def tts(text: str, voice: str = VOICE) -> Generator[bytes, None, None]:
    """Edge TTS (스트리밍)"""
    try:
        communicate = Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS 오류: {e}") from e

# --- FastAPI 엔드포인트 ---
@app.post("/tts_stream")
async def text_to_speech_stream(request_data: dict) -> StreamingResponse:
    """텍스트 음성 변환 (스트리밍)"""
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return StreamingResponse(tts(text), media_type="audio/mpeg")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 통신"""
    await websocket.accept()
    context: List[int] = []
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("text")
            image_data = data.get("image")

            if user_input or image_data:
                ollama_response = await query_ollama(
                    prompt=f"{SYSTEM_PROMPT}\n\n[INST] text: {user_input if user_input else ''} [/INST]",
                    image_data=image_data,
                    context=context,
                )
                if "error" in ollama_response:
                    await websocket.send_json({"error": ollama_response["error"]})
                    continue

                response_json = ollama_response
                try:
                    response_text = response_json["response"]

                    new_context = ollama_response.get('context', [])
                    if isinstance(new_context, list) and all(isinstance(item, int) for item in new_context):
                         context = new_context
                    else:
                        logger.warning(f"Invalid context: {new_context}")

                    await websocket.send_json({"text": response_text, "json_data": response_json})

                except KeyError as e:
                     logger.error(f"KeyError: {e}.  Ollama response: {ollama_response}")
                     await websocket.send_json({"error": f"Ollama response missing 'response' key: {ollama_response}"})
                except Exception as e:
                    logger.exception(f"Error processing Ollama response: {ollama_response}")
                    await websocket.send_json({"error": "Error processing Ollama response"})
            else:
                await websocket.send_json({"error": "Invalid input"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        await websocket.send_json({"error": str(e)})

# --- 정적 파일 제공 ---
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
