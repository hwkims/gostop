![image](https://github.com/user-attachments/assets/abfc0341-2bb6-40d5-900c-0b99c6ee93a1)
![image](https://github.com/user-attachments/assets/200b5d56-9b72-4bd5-9ffc-25c16592a8c5)
![image](https://github.com/user-attachments/assets/8512c831-2c3e-41f6-9054-deb97045ffba)
![image](https://github.com/user-attachments/assets/cd1866ef-00e7-40e0-8be3-bb09d6f7c20f)

# 🚦 GoStop - 이미지 기반 직진/정지 판단 보조 도구

이 프로젝트는 [Streamlit](https://streamlit.io/)으로 구축된 간단한 웹 애플리케이션으로, 로컬에서 실행 중인 [Ollama](https://ollama.com/) 인스턴스와 `granite3.2-vision` 모델을 사용하여 업로드된 이미지를 분석합니다. 분석 결과를 바탕으로 해당 이미지가 '직진(Go)'해야 할 상황인지 '정지(Stop)'해야 할 상황인지를 판단하는 데 도움을 주는 것을 목표로 합니다.

**주의:** 이 도구는 실험적인 목적으로 제작되었으며, 실제 운전 결정에 사용되어서는 **절대 안 됩니다**. 모델의 판단은 정확하지 않을 수 있습니다.

## ✨ 주요 기능

*   간단한 웹 인터페이스를 통한 이미지 업로드
*   로컬 Ollama API (`granite3.2-vision` 모델) 와의 연동
*   업로드된 이미지에 대한 모델의 텍스트 기반 분석 결과 표시
*   모델 응답에서 '직진' 또는 '정지' 키워드를 감지하여 간단한 시각적 피드백 제공 (🔴 정지 / 🟢 직진)

## 🤔 작동 방식

1.  사용자가 Streamlit 웹 인터페이스를 통해 이미지 파일을 업로드합니다.
2.  업로드된 이미지는 Base64로 인코딩되어 로컬 Ollama API (`/api/generate` 엔드포인트)로 전송됩니다.
3.  이미지와 함께 특정 프롬프트("이 이미지를 보고 ... '직진'해야 할지 '정지'해야 할지 판단해주세요...")가 `granite3.2-vision` 모델로 전달됩니다.
4.  Ollama는 이미지를 분석하고 모델의 텍스트 응답을 반환합니다.
5.  Streamlit 앱은 모델이 생성한 전체 텍스트 응답을 표시하고, 응답 내용에 '직진' 또는 '정지' 키워드가 포함되어 있는지 여부에 따라 간단한 해석(🔴/🟢)을 덧붙여 보여줍니다.

## ⚙️ 사전 요구 사항

이 애플리케이션을 실행하기 전에 다음 사항들이 준비되어 있어야 합니다:

1.  **Python:** 버전 3.8 이상 권장
2.  **Ollama:** 로컬 컴퓨터에 [Ollama가 설치](https://ollama.com/)되어 실행 중이어야 합니다.
    *   백그라운드 실행: 터미널에서 `ollama serve` 명령 실행
3.  **Ollama Model (`granite3.2-vision`):** 필요한 모델을 미리 다운로드해야 합니다.
    ```bash
    ollama pull granite3.2-vision
    ```
4.  **Python 라이브러리:** `streamlit`과 `requests` 라이브러리가 필요합니다.

## 🚀 설치 및 실행

1.  **저장소 복제:**
    ```bash
    git clone https://github.com/hwkims/gostop.git
    cd gostop
    ```

2.  **필요한 라이브러리 설치:**
    ```bash
    pip install streamlit requests
    ```
    *(선택 사항: `requirements.txt` 파일을 생성하여 관리할 수도 있습니다.)*

3.  **Ollama 서버 실행 확인:**
    별도의 터미널에서 `ollama serve`를 실행하거나 Ollama 애플리케이션이 백그라운드에서 실행 중인지 확인합니다.

4.  **Streamlit 앱 실행:**
    ```bash
    streamlit run app.py
    ```
    *(Python 스크립트 파일 이름이 `app.py`라고 가정합니다.)*

5.  **웹 브라우저 접속:**
    앱이 실행되면 터미널에 표시된 로컬 URL(일반적으로 `http://localhost:8501`)을 웹 브라우저에서 엽니다.

6.  **이미지 분석:**
    웹 페이지의 안내에 따라 분석할 이미지 파일을 업로드하고 "이미지 분석 시작" 버튼을 클릭합니다.

## 🛠️ 기술 스택

*   **언어:** Python
*   **웹 프레임워크:** Streamlit
*   **HTTP 요청:** Requests
*   **LLM 실행 환경:** Ollama
*   **Vision-Language 모델:** IBM Granite-3.2-Vision (`granite3.2-vision`)

## ⚠️ 중요 참고 사항 및 한계점

*   **모델 특성:** `granite3.2-vision` 모델은 주로 **문서 이해(표, 차트, 다이어그램 등)**에 특화되어 학습되었습니다. 따라서 일반적인 교통 상황, 신호등, 표지판 이미지에 대한 해석 능력은 범용 Vision 모델이나 교통 상황에 특화된 모델보다 제한적일 수 있습니다. 결과가 부정확하거나 예상과 다를 수 있습니다.
*   **프롬프트 의존성:** 분석 결과는 모델에게 전달하는 프롬프트에 크게 의존합니다. `app.py` 코드 내의 프롬프트를 수정하여 더 나은 결과를 유도해 볼 수 있습니다.
*   **로컬 환경:** 이 애플리케이션은 전적으로 사용자의 로컬 컴퓨터에서 실행되는 Ollama에 의존합니다. 외부 클라우드 API를 사용하지 않습니다.
*   **리소스 사용량:** Ollama와 Vision 모델을 실행하는 것은 상당한 CPU/GPU 및 메모리(RAM) 리소스를 소모할 수 있습니다. 시스템 사양에 따라 실행 속도가 느릴 수 있습니다.
*   **실사용 불가:** **이 프로젝트는 기술 데모 및 실험용이며, 실제 운전 상황에서의 의사 결정에 절대로 사용해서는 안 됩니다.** 생명이나 안전과 관련된 결정에는 신뢰할 수 있는 정보와 운전자 본인의 판단이 필수적입니다.

## 📄 라이선스

[  MIT License, Apache License 2.0 등] 
