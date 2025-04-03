import streamlit as st
import requests
import base64
import io
import json

# --- 설정 ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API 엔드포인트
MODEL_NAME = "granite3.2-vision" # 사용할 모델 이름

# --- Ollama API 호출 함수 ---
def get_ollama_vision_response(image_bytes, prompt):
    """
    Ollama Vision 모델 API를 호출하고 응답을 반환합니다.
    """
    try:
        # 이미지를 Base64로 인코딩
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False, # 스트리밍 없이 한 번에 응답 받기
            "options": {
                "temperature": 0 # 예측 가능성을 높이기 위해 temperature 0 사용
            }
        }

        headers = {'Content-Type': 'application/json'}

        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload), timeout=120) # 타임아웃 늘리기
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생

        # 응답 JSON 파싱
        data = response.json()
        return data.get('response', '모델 응답에서 "response" 키를 찾을 수 없습니다.')

    except requests.exceptions.ConnectionError:
        return "오류: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요."
    except requests.exceptions.Timeout:
        return "오류: Ollama 서버 응답 시간이 초과되었습니다."
    except requests.exceptions.RequestException as e:
        return f"Ollama API 요청 중 오류 발생: {e}\n응답 내용: {response.text if 'response' in locals() else 'N/A'}"
    except Exception as e:
        return f"알 수 없는 오류 발생: {e}"

# --- Streamlit 앱 UI ---
st.set_page_config(layout="wide")
st.title(f"🚦 이미지 기반 직진/정지 판단 ({MODEL_NAME})")
st.write("이미지를 업로드하면 Ollama Vision 모델이 분석하여 '직진' 또는 '정지' 여부를 판단합니다.")

uploaded_file = st.file_uploader("분석할 이미지 파일을 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 바이트 읽기
    image_bytes = uploaded_file.getvalue()

    # 이미지 표시 (가운데 정렬 시도)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image_bytes, caption='업로드된 이미지', use_column_width=True)

    # 분석 버튼
    if st.button("🚀 이미지 분석 시작"):
        # 사용자에게 전달할 프롬프트 정의
        # 모델이 '직진' 또는 '정지' 라는 단어를 포함하여 답변하도록 유도
        prompt = """이 이미지를 보고 교통 상황이나 표지판 등을 분석해서, 지금 '직진'해야 할 상황인지 '정지'해야 할 상황인지 판단해주세요. 답변은 '직진' 또는 '정지' 라는 단어를 반드시 포함하고, 이유를 간략하게 설명해주세요."""

        st.info(f"'{MODEL_NAME}' 모델에게 다음 프롬프트로 질문합니다:\n'{prompt}'")

        # Ollama API 호출 및 결과 표시
        with st.spinner('이미지를 분석 중입니다...'):
            result = get_ollama_vision_response(image_bytes, prompt)

            st.subheader("💡 분석 결과")
            st.write(result)

            # 간단한 키워드 기반 판단 (선택 사항)
            if "정지" in result and "직진" not in result:
                st.error("🔴 정지 신호 또는 상황으로 판단됩니다.")
            elif "직진" in result and "정지" not in result:
                st.success("🟢 직진 가능 상황으로 판단됩니다.")
            elif "정지" in result and "직진" in result:
                 st.warning("⚠️ 직진과 정지 키워드가 모두 포함되어 있습니다. 내용을 직접 확인하세요.")
            else:
                st.warning("결과에서 '직진' 또는 '정지' 키워드를 명확히 찾지 못했습니다. 전체 응답을 참고하세요.")

else:
    st.info("분석할 이미지를 업로드해주세요.")

st.sidebar.header("정보")
st.sidebar.markdown(f"""
이 앱은 로컬에서 실행 중인 Ollama 서버의 `{MODEL_NAME}` 모델을 사용합니다.

**요구 사항:**
- Ollama 설치 및 실행
- `{MODEL_NAME}` 모델 다운로드 (`ollama pull {MODEL_NAME}`)
- Python `streamlit`, `requests` 라이브러리

**Ollama API:** `{OLLAMA_API_URL}`
""")
