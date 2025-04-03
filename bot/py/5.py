from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64
import json
import io
import logging
import asyncio
from typing import Optional, Generator, List, Dict, Any
from pydantic import BaseModel
from edge_tts import Communicate
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama 서버 주소 및 음성 설정 (환경 변수 사용 권장)
OLLAMA_HOST = "http://localhost:11434"  # Ollama 기본 주소
VOICE = "ko-KR-HyunsuNeural"
MODEL_NAME = "gemma3:4b" # 4b로 고정

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (개발용), 실제 환경에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 시스템 프롬프트 (개선됨)
SYSTEM_PROMPT = """
너는 시각 장애인을 위한 안내견 로봇, 범블비야. 너의 임무는 사용자를 안전하게 안내하고, 사용자의 질문에 친절하고 정확하게 답변하며, 쇼핑을 도와주는 거야. 너는 과거의 경험을 JSON 파일('memory.json')에 저장하고, 필요할 때 이 메모리를 참고하여 더 나은 결정을 내릴 수 있어.

로봇의 카메라에서 캡처한 이미지와 사용자가 텍스트로 입력한 명령/질문을 받으면, 다음 작업을 **단계별로** 수행해야 해:

1.  **JSON 메모리 읽기:** 'memory.json' 파일에서 최근 50개 이벤트를 읽어서 과거의 경험을 확인해.  (구현 필요)
2.  **이미지 분석:**
    *   이미지에 있는 물체들을 식별하고, 각 물체의 상대적인 위치를 12시 방향 기준으로 파악해.
    *   문이 있으면, 문 손잡이와 경첩의 위치를 파악하고, 문에 "당기시오" 또는 "미시오"와 같은 텍스트가 있는지 확인해. 문의 여닫는 방향을 경첩의 위치로 파악해야해.
    *   상품이 있으면, 상품의 이름, 가격, 특징 등을 파악해. (OCR, 외부 API 연동)
    *   장면을 상세하게 설명해.
    *   로봇의 현재 상황(현재 위치, 주변 환경, 장애물 유무 등)을 파악해.
3.  **명령/질문 해석:** 사용자가 무엇을 원하는지, 어떤 정보를 필요로 하는지 정확하게 이해해.
4.  **행동 계획 (Chain-of-Thought):**
    *   명령을 수행하거나 질문에 답변하기 위해 필요한 **단계별** 행동 계획을 세워.
    *   각 단계에서 수행할 동작, 필요한 센서 정보, 예상되는 결과, 고려해야 할 사항(예: 안전) 등을 **자세하게** 설명해.
    *   "안 된다"는 말은 최대한 자제하고, 주어진 능력(바퀴)을 최대한 활용해서 임무를 완수할 방법을 찾아봐.
    *   필요하다면 조건부 동작(if-else), 반복 동작(while, for), 시간 지연(wait) 등을 사용해.
5.  **JSON 출력:** ROS 2 토픽(topic)을 통해 로봇에게 전달할 메시지를 JSON 형식으로 출력해. `motor_commands`는 `geometry_msgs/Twist` 메시지 형식과 호환되어야 해.
6.  **JSON 메모리 업데이트:** 현재 상황, 사용자 명령/질문, 수행한 작업, 결과, 생성한 JSON 출력, `speech_output`을 JSON 파일에 새로운 이벤트로 추가해. (구현 필요)

JSON 출력에는 다음 키들이 포함되어야 해:

*   `scene_description`: (문자열) 카메라에 찍힌 장면을 상세하게 설명.
*   `objects_detected`: (문자열 리스트) 이미지에서 감지된 물체들의 목록.
*    `object_locations`: (객체 리스트) 이미지에서 감지된 객체들의 위치
    *   `object`:(문자열) 객체의 이름
    *   `relative_position`: (문자열) 객체의 상대적 위치 (예: "정면 50cm", "왼쪽", "테이블 위")
     *   `clock_direction`: (문자열) 12시 방향을 기준으로한 객체의 방향
*   `object_details`: (객체 리스트, 선택 사항) 감지된 객체에 대한 추가 정보
    *   `object`: (문자열) 객체 이름
    *   `details`: (딕셔너리) 객체에 대한 추가 정보 (키-값 쌍)
        *   `color`: (문자열)
        *   `size`: (문자열)
        *   `material`: (문자열)
        *   `status`: (문자열)
        *   `price`: (실수, 선택 사항)
        *   `features`: (문자열 리스트, 선택 사항)
*   `command_interpretation`: (문자열, 선택 사항) 사용자의 명령/질문을 명확하게 풀어서 설명.
*   `action_plan`: (문자열 리스트) 로봇이 수행할 행동을 단계별로 설명.
*  `plan_explanation`: (문자열) : 행동계획에 대한 부가 설명
*   `action_steps`: (객체 리스트) 로봇이 수행할 단계를 자세하게 정의.
    *   `step_id`: (정수) 단계 번호.
    *   `description`: (문자열) 단계에 대한 설명.
    *   `type`: (문자열) 단계 유형 ("move", "turn", "wait", "check_sensor", "speak", "other").
    *   `duration`: (실수, 선택 사항) 단계 지속 시간 (초).
    *   `condition`: (문자열, 선택 사항) 조건부 동작의 조건.
    *   `sensor_type`: (문자열, `check_sensor` 단계에서 사용) 확인할 센서 종류 ("camera").
    *   `expected_value`: (문자열, `check_sensor` 단계에서 사용) 기대하는 센서 값 또는 상태 (예: "obstacle_detected").
    *   `motor_commands`: (딕셔너리, 선택 사항) ROS 2 `geometry_msgs/Twist` 메시지.
        *   `linear`: (딕셔너리)
            *   `x`: (실수) 전/후 방향 속도 (m/s)
            *   `y`: (실수)
            *   `z`: (실수)
        *   `angular`: (딕셔너리)
            *   `x`: (실수)
            *   `y`: (실수)
            *   `z`: (실수) 좌/우 회전 속도 (rad/s)
    *   `repeat`: (정수, 선택 사항) 반복 횟수.
    *  `distance` : (선택사항, 실수) 이동거리
    *  `angle`: (선택사항, 실수) 회전 각도
*   `is_safe`: (boolean) 현재 상황이 안전한지 여부
*   `speech_output`: (문자열) 로봇이 사용자에게 말할 내용.

**중요 사항:**

*   JSON 메모리에서 현재 상황과 관련된 정보를 찾아 활용해.
*   설명과 지시는 간결하고 구체적으로 작성해.
*   상황에 맞는 재치있고 친절한 말투를 사용하고, 유머나 감정 표현도 적절히 섞어줘. 범블비처럼!
"""

class OllamaRequest(BaseModel):
    """Ollama API 요청 데이터 모델"""
    prompt: str
    image: Optional[str] = None
    context: Optional[List[int]] = None  # context는 정수 리스트

def query_ollama(prompt: str, image_data: Optional[str] = None, context: Optional[List[int]] = None) -> Dict[str, Any]:
    """Ollama API 호출 함수 (타임아웃, 재시도 로직 추가)"""
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "context": context or [],
        "options": {"temperature": 0.2, "top_p": 0.8},
    }
    if image_data:
        data["images"] = [image_data]

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=data,
                timeout=60,  # 타임아웃 60초
            )
            response.raise_for_status()  # 200 OK 외에는 예외 발생
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama API 요청 오류 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Ollama API 요청 실패: {e}")
            asyncio.sleep(retry_delay)  # 재시도 전 대기
        except json.JSONDecodeError as e:
            logger.error(f"Ollama API 응답 JSON 디코딩 오류: {e}")
            raise HTTPException(status_code=500, detail=f"Ollama API 응답 JSON 디코딩 오류: {e}")

async def tts(text: str, voice: str = VOICE) -> Generator[bytes, None, None]:
    """Edge TTS를 사용한 텍스트 음성 변환 (스트리밍)"""
    try:
        communicate = Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    except Exception as e:
        logger.error(f"TTS 오류: {e}")
        raise HTTPException(status_code=500, detail=f"TTS 오류: {e}")

# --- FastAPI 엔드포인트 ---

@app.post("/generate")
async def generate_response(request_data: OllamaRequest) -> Dict[str, Any]:
    """Ollama API를 호출하여 응답 생성"""
    try:
        ollama_response = query_ollama(
            prompt=f"{SYSTEM_PROMPT}\n\n[INST] text: {request_data.prompt} [/INST]",
            image_data=request_data.image,
            context=request_data.context,
        )
        return ollama_response

    except HTTPException as e: #HTTPException 처리
        raise e
    except Exception as e:
        logger.exception("예상치 못한 오류 in /generate")  # 더 자세한 로깅
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류: {e}")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)) -> Dict[str, str]:
    """이미지 파일을 받아 base64로 인코딩하여 반환"""
    try:
        contents = await file.read()
        img_base64 = base64.b64encode(contents).decode("utf-8")
        return {"image_base64": img_base64}
    except Exception as e:
        logger.error(f"이미지 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 오류: {e}")

@app.post("/tts_stream")
async def text_to_speech_stream(request_data: dict) -> StreamingResponse:
    """텍스트를 음성으로 변환 (스트리밍)"""
    text = request_data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return StreamingResponse(tts(text), media_type="audio/mpeg")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 통신 처리"""
    await websocket.accept()
    context: List[int] = [] #Ollama context
    try:
        while True:
            data = await websocket.receive_json()
            user_input = data.get("text")
            image_data = data.get("image")

            if user_input or image_data:
                ollama_response = query_ollama(
                    prompt=f"{SYSTEM_PROMPT}\n\n[INST] text: {user_input if user_input else ''} [/INST]",
                    image_data=image_data,
                    context=context,
                )

                if "error" in ollama_response:
                    await websocket.send_json({"error": ollama_response["error"]})
                    continue

                response_json = ollama_response
                try:
                    response_text = response_json["response"] #response 키 사용

                    # context 업데이트 (정수 리스트인지 확인)
                    new_context = ollama_response.get('context', [])
                    if isinstance(new_context, list) and all(isinstance(item, int) for item in new_context):
                        context = new_context
                    else:
                        logger.warning(f"Invalid context received: {new_context}")
                    await websocket.send_json({"text": response_text, "json_data": response_json})
                except KeyError:
                     logger.error(f"KeyError: Ollama response: {ollama_response}")
                     await websocket.send_json({"error": f"Ollama 응답에 'response' 키가 없습니다: {ollama_response}"})
                except Exception as e:  # JSON 파싱 오류 처리
                    logger.exception(f"Ollama response parsing error: {ollama_response}")
                    await websocket.send_json({"error": "Ollama 응답 처리 중 오류 발생"})
            else:
                await websocket.send_json({"error": "Invalid input: text or image required"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        await websocket.send_json({"error": str(e)})

# 정적 파일 제공 (index.html, CSS, JS 등)
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """index.html 파일 제공"""
    try:
        with open(static_dir / "index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
