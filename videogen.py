import asyncio
import os
import re
import shutil
import ffmpeg
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
import textwrap
from bing_image_downloader import downloader
import edge_tts

async def generate_tts(text, voice="ko-KR-HyunsuNeural", output_file="temp.mp3"):
    """Generate TTS audio using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def create_video_from_script(script_tsv, output_filename="tutorial_video.mp4"):
    """Create a video from a TSV script with one sentence per scene."""
    width, height = 1920, 1080
    font_size = 60
    sentence_num = 0

    # Parse TSV
    lines = script_tsv.strip().split("\n")
    script_data = []
    for line in lines[1:]:  # Skip header
        if line.strip():
            parts = line.split("\t")
            if len(parts) == 2:
                keyword, content = parts
                script_data.append((keyword.strip(), content.strip()))
            else:
                print(f"Skipping invalid line: {line}")

    concat_file_path = "mylist.txt"
    with open(concat_file_path, "w") as f:
        pass

    for keyword, content in script_data:
        sentence_num += 1
        print(f"Processing sentence {sentence_num}...")

        # Download background image
        downloader.download(
            keyword, limit=1, output_dir="image", adult_filter_off=False,
            force_replace=False, timeout=60, verbose=False
        )
        image_dir = os.path.join("image", keyword)
        image_clip_path = f"image_clip_{sentence_num}.png"

        try:
            image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            if not image_files:
                raise FileNotFoundError(f"No image found for {keyword}")
            image_path = os.path.join(image_dir, image_files[0])
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((width, height), Image.Resampling.LANCZOS)
            background = Image.new("RGB", (width, height), "black")
            img_w, img_h = img.size
            offset = ((width - img_w) // 2, (height - img_h) // 2)
            background.paste(img, offset)
            background.save(image_clip_path)
        except Exception as e:
            print(f"Image error for {keyword}: {e}. Using black background.")
            background = Image.new("RGB", (width, height), "black")
            background.save(image_clip_path)

        # Generate TTS
        tts_filename = f"tts_{sentence_num}.mp3"
        asyncio.run(generate_tts(content, output_file=tts_filename))
        audio = AudioSegment.from_file(tts_filename)
        duration = audio.duration_seconds

        # Create text overlay
        text_clip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(text_clip)
        try:
            font = ImageFont.truetype("BMJUA_otf.otf", font_size)
        except IOError:
            print("Font BMJUA_otf.otf not found. Using default.")
            font = ImageFont.load_default()

        lines = textwrap.wrap(content, width=45)
        y_text = height - (len(lines) * (font_size * 1.2)) - 100
        for line in lines:
            if line:
                bbox = d.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (width - text_width) / 2
                y = y_text
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        d.text((x + i, y + j), line, font=font, fill="white")
                d.text((x, y), line, font=font, fill="black")
                y_text += font_size * 1.2

        text_clip_path = f"text_clip_{sentence_num}.png"
        text_clip.save(text_clip_path)

        # Combine image, text, and audio
        output_video = f"output_{sentence_num}.mp4"
        try:
            input_image = ffmpeg.input(image_clip_path, loop=1, framerate=24, t=duration)
            input_audio = ffmpeg.input(tts_filename)
            input_text = ffmpeg.input(text_clip_path, loop=1, framerate=24, t=duration)
            overlay_layer = input_image.overlay(input_text)
            (
                ffmpeg.concat(overlay_layer, input_audio, v=1, a=1)
                .output(output_video, vcodec="libx264", acodec="aac", pix_fmt="yuv420p")
                .run(overwrite_output=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return

        with open(concat_file_path, "a") as f:
            f.write(f"file '{output_video}'\n")

    # Concatenate all clips
    try:
        ffmpeg.input(concat_file_path, format="concat", safe=0).output(
            output_filename, c="copy"
        ).run(overwrite_output=True, quiet=True)
        print(f"Video created: {output_filename}")
    except ffmpeg.Error as e:
        print(f"Concatenation error: {e.stderr.decode()}")
        return

    # Clean up
    for i in range(1, sentence_num + 1):
        for file in [f"tts_{i}.mp3", f"text_clip_{i}.png", f"output_{i}.mp4", f"image_clip_{i}.png"]:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except OSError as e:
                print(f"Error deleting {file}: {e}")
    try:
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
        if os.path.exists("image"):
            shutil.rmtree("image")
    except OSError as e:
        print(f"Cleanup error: {e}")

if __name__ == "__main__":
    script_tsv = """
keyword	content
JetBot	"안녕하세요!"
JetBot	"오늘은 인공지능의 눈과 귀를 가진 로봇, JetBot을 만들어 볼 겁니다!"
Ollama	"JetBot과 Ollama를 활용해서 뭘 할까요?"
FastAPI	"FastAPI로 원격 제어를 구현할 거예요."
Edge TTS	"그리고 Edge TTS로 실시간 상황을 음성으로 들을 수 있어요."
Jetson Nano	"젯슨 나노 기반의 JetBot이 카메라로 주변을 봅니다."
Ollama	"Ollama가 그걸 분석해요."
FastAPI	"FastAPI 서버가 PC로 데이터를 전달하죠."
Edge TTS	"Edge TTS가 음성으로 상황을 설명해 줍니다!"
Coding Tutorial	"코딩 초보도 따라 할 수 있어요."
JetBot Web Interface	"먼저 완성된 프로젝트를 보여드릴게요."
Webcam	"이건 JetBot 제어 센터예요."
Webcam	"카메라로 실시간 주변을 볼 수 있죠."
Control Buttons	"전진 버튼을 누르면요?"
Control Buttons	"JetBot이 앞으로 가요!"
Control Buttons	"후진, 좌회전, 우회전도 가능해요."
Control Buttons	"정지 버튼도 있죠."
Ollama Response	"Ollama가 분석한 내용은 여기 텍스트로 나와요."
Edge TTS	"Edge TTS가 그걸 음성으로 읽어줍니다."
Demo	"‘주변 설명’ 버튼을 눌러볼까요?"
Demo	"JetBot이 주변 상황을 말해줘요."
Demo	"‘앞으로’ 버튼을 누르면 전진하고요."
Demo	"‘장애물 피하기’를 실행하면요?"
Demo	"JetBot이 장애물을 피해가요!"
JetBot Components	"이 프로젝트는 5가지 구성 요소로 되어 있어요."
JetBot	"먼저, 주인공 JetBot이에요."
Jetson Nano	"Jetson Nano가 두뇌 역할을 하죠."
PC	"PC에서 FastAPI 서버를 돌려요."
Ollama	"Ollama는 로컬에서 언어 모델을 실행해요."
FastAPI	"FastAPI는 JetBot과 PC를 연결하죠."
Edge TTS	"Edge TTS는 음성을 만들어줘요."
Diagram	"이 구성 요소들이 어떻게 연결될까요?"
Diagram	"JetBot 카메라가 데이터를 보내요."
Diagram	"Ollama가 분석하고요."
Diagram	"FastAPI가 PC로 전달해요."
Diagram	"마지막으로 TTS가 소리로 바꿔줍니다."
Installation	"이제 설치를 시작해 볼까요?"
JetBot Setup	"JetBot과 Jetson Nano는 준비됐다고 가정할게요."
Python	"PC에 Python이 설치되어 있어야 해요."
Terminal	"Ollama를 먼저 설치합시다."
Terminal	"터미널에서 ollama run gemma3:4b를 실행해요."
Code Editor	"이제 프로젝트 코드를 다운로드 받아요."
Terminal	"pip install -r requirements.txt로 패키지를 설치해요."
Code Editor	"main.py 파일을 열어볼까요?"
Code Editor	"JETBOT_WEBSOCKET_URL을 수정해야 해요."
Terminal	"FastAPI 서버를 실행합시다."
Web Browser	"브라우저에서 localhost:8000에 접속하면요?"
Web Interface	"JetBot 제어 센터가 나타나요!"
FastAPI Code	"이제 main.py 코드를 볼게요."
Pydantic	"OllamaRequest 모델이 요청 데이터를 정의해요."
Ollama API	"query_ollama_stream 함수가 Ollama와 통신해요."
WebSocket	"send_command_to_jetbot이 JetBot에 명령을 보내죠."
TTS Function	"speak 함수가 음성을 생성해요."
Generate Function	"generate 함수가 모든 걸 처리합니다."
JetBot Code	"이제 jetbot_control.py를 볼게요."
Handle Command	"handle_command 함수가 JetBot을 움직여요."
WebSocket	"websocket_handler가 연결을 관리하고요."
Camera	"send_images가 카메라 이미지를 보내줘요."
Features	"추가 기능도 넣을 수 있어요!"
Face Recognition	"얼굴 인식을 추가할까요?"
Object Tracking	"물체를 따라다니게 할 수도 있고요."
SLAM	"SLAM으로 지도를 만들 수도 있어요."
Voice Command	"음성 명령도 가능해요!"
Conclusion	"오늘 내용은 여기까지예요."
Conclusion	"JetBot 시스템을 함께 만들어봤는데요."
Conclusion	"어떠셨나요?"
Conclusion	"궁금한 건 댓글로 물어보세요!"
Conclusion	"구독과 좋아요도 부탁드려요!"
"""
    create_video_from_script(script_tsv, output_filename="jetbot_tutorial.mp4")
