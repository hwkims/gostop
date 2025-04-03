# -*- coding: utf-8 -*-
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import base64
import binascii
from typing import Dict, List, Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager
import httpx
from pathlib import Path
import edge_tts
import os
import websockets
import re

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "granite3.2-vision"  # Vision 모델 확인 필요
DRIVER_WEBSOCKET_URL = "ws://192.168.137.181:8766"  # JetBot IP 확인
STATIC_DIR = Path(__file__).parent / "static"
TTS_VOICE = "en-US-JennyNeural"
VALID_COMMANDS = [
    "forward_slow", "forward_medium", "forward_fast",
    "backward_slow", "backward_medium", "backward_fast",
    "left_slow", "left_medium", "left_fast",
    "right_slow", "right_medium", "right_fast",
    "stop"
]
HTTP_TIMEOUT = 30.0
MIN_IMAGE_LENGTH = 1000
AUTONOMOUS_INTERVAL = 0.2  # 자율주행 주기 (초)

OLLAMA_OPTIONS = {"temperature": 0.0}  # 일관된 결정

# --- Global State ---
client_websockets: List[WebSocket] = []
driver_websocket: Optional[websockets.WebSocketClientProtocol] = None
current_image_base64: Optional[str] = None
autonomous_task: Optional[asyncio.Task] = None
is_autonomous_running: bool = False
last_command: str = "stop"
driver_connection_task: Optional[asyncio.Task] = None

# --- FastAPI Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global driver_connection_task
    logger.info("Application startup...")
    driver_connection_task = asyncio.create_task(connect_to_driver())
    yield
    logger.info("Application shutting down...")
    if is_autonomous_running: await stop_autonomous_mode()
    if driver_connection_task and not driver_connection_task.done():
        driver_connection_task.cancel(); await asyncio.sleep(0.1)
    if driver_websocket:
        try: await driver_websocket.close(code=1001)
        except Exception: pass
    clients_to_close = client_websockets[:]
    for ws in clients_to_close:
        try: await ws.close(code=1001)
        except Exception: pass
    client_websockets.clear()
    logger.info("Shutdown complete.")

# --- FastAPI App Setup ---
app = FastAPI(title="JetBot Vision Control v25 (Road Autonomous)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Helper Functions ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index_0326_1.html"
    if not index_path.exists(): return HTMLResponse("Error: index_0326_1.html not found", 404)
    try:
        with open(index_path, "r", encoding="utf-8") as f: return HTMLResponse(f.read())
    except Exception as e: logger.error(f"Read index.html error: {e}"); return HTMLResponse("Server error", 500)

async def generate_tts(text: str) -> Optional[str]:
    if not text: return None
    uid = base64.urlsafe_b64encode(os.urandom(9)).decode('utf-8')
    tmp_file = STATIC_DIR / f"tts_{uid}.mp3"
    try:
        comm = edge_tts.Communicate(text, TTS_VOICE); await comm.save(str(tmp_file))
        if not tmp_file.exists(): raise IOError("TTS save failed")
        with open(tmp_file, "rb") as f: audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        return audio_b64
    except Exception as e: logger.error(f"TTS gen error: {e}"); return None
    finally:
        if tmp_file.exists():
            try: os.remove(tmp_file)
            except OSError: pass

def strip_base64_prefix(b64_str: str) -> str:
    return re.sub(r"^data:image/[a-zA-Z]+;base64,", "", b64_str, flags=re.IGNORECASE)

def is_base64_valid(b64_string: Optional[str]) -> bool:
    if not b64_string or len(b64_string) < MIN_IMAGE_LENGTH: return False
    try: base64.b64decode(b64_string, validate=True); return True
    except (binascii.Error, ValueError): return False

# --- Ollama Interaction ---
async def query_ollama_describe_ui(image_data: str) -> Dict:
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string): return {"description": "Error: Invalid image data."}
    prompt_text = (
        "As JetBot's camera observer on a road, describe the scene factually and concisely. "
        "Focus on road conditions, traffic signals, obstacles, and path curvature relevant to driving. "
        "Respond **strictly** JSON: {\"description\": \"<scene description>\"}"
    )
    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            if not inner_json_str: return {"description": "Empty response."}
            result = json.loads(inner_json_str)
            desc = str(result.get("description", "")).strip() or "No description."
            return {"description": desc}
    except Exception as e: logger.error(f"Ollama describe error: {e}"); return {"description": "Error during description."}

async def query_ollama_autonomous(image_data: str, last_cmd: str = "stop") -> Dict[str, str]:
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string):
        logger.error("Invalid image for autonomous decision. Stopping.")
        return {"command": "stop"}

    valid_cmds_str = ", ".join(VALID_COMMANDS)
    prompt_text = (
        f"**Act as JetBot's autonomous driver on a road.** Last command: '{last_cmd}'. Analyze the camera image and select ONE command from: {valid_cmds_str}. "
        "Prioritize safety and road rules. Follow these steps strictly:\n\n"
        "**1. Safety Check (STOP Conditions):**"
        "\n- RED traffic light visible? -> 'stop'"
        "\n- Obstacle (vehicle, pedestrian, object) < 1m ahead? -> 'stop'"
        "\n- Large obstacle blocking path < 3m? -> 'stop'"
        "\n- Road ends or path blocked? -> 'stop'"
        "\n- If ANY stop condition is true, output 'stop' and skip to response.\n\n"
        "**2. Road Analysis (If NO Stop):**"
        "\n- YELLOW light? -> 'forward_slow'"
        "\n- GREEN light or clear straight road:"
        "    - >10m clear? -> 'forward_fast'"
        "    - 5-10m clear? -> 'forward_medium'"
        "    - <5m clear? -> 'forward_slow'"
        "\n- Curve ahead:"
        "    - Gentle curve? -> 'left_slow' or 'right_slow'"
        "    - Moderate curve? -> 'left_medium' or 'right_medium'"
        "    - Sharp curve? -> 'left_fast' or 'right_fast'"
        "\n- Obstacle avoidance (if clear to maneuver):"
        "    - Obstacle on right? -> 'left_slow'"
        "    - Obstacle on left? -> 'right_slow'"
        "\n- If uncertain or no safe path, default to 'stop'.\n\n"
        "**Respond ONLY:** {\"command\": \"<chosen_command>\"}"
    )
    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            if not inner_json_str: logger.warning("Ollama empty response. Stopping."); return {"command": "stop"}
            result = json.loads(inner_json_str)
            command = result.get("command", "stop")
            if command not in VALID_COMMANDS: logger.warning(f"Invalid command: '{command}'. Stopping."); command = "stop"
            return {"command": command}
    except Exception as e:
        logger.error(f"Ollama autonomous query failed: {e}")
        return {"command": "stop"}

async def query_ollama_interpret_custom(image_data: str, custom_prompt: str) -> Dict[str, str]:
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string):
        logger.error("Invalid image for custom command. Stopping.")
        return {"command": "stop"}
    valid_cmds_str = ", ".join(VALID_COMMANDS)
    prompt_text = (
        f"**JetBot command interpreter for road driving.** User request: '{custom_prompt}'. "
        f"Analyze the image and choose ONE safe command from: {valid_cmds_str}. "
        "Ensure it matches the request and road conditions (e.g., stop at red light, avoid obstacles). "
        "If unsafe or unclear, choose 'stop'. Respond ONLY: {\"command\": \"<chosen_command>\"}"
    )
    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            if not inner_json_str: logger.warning("Ollama empty custom response. Stopping."); return {"command": "stop"}
            result = json.loads(inner_json_str)
            command = result.get("command", "stop")
            if command not in VALID_COMMANDS: logger.warning(f"Invalid custom command: '{command}'. Stopping."); command = "stop"
            return {"command": command}
    except Exception as e:
        logger.error(f"Ollama custom query failed: {e}")
        return {"command": "stop"}

# --- WebSocket Handling ---
async def connect_to_driver():
    global driver_websocket, current_image_base64
    while True:
        ws_url = DRIVER_WEBSOCKET_URL
        try:
            async with websockets.connect(ws_url, ping_interval=10, ping_timeout=20, open_timeout=10) as ws:
                driver_websocket = ws; logger.info(f"Driver connected: {ws_url}.")
                await broadcast_to_clients({"status": "driver_connected"})
                while True:
                    try: data = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        try: await asyncio.wait_for(ws.ping(), timeout=10); continue
                        except asyncio.TimeoutError: logger.error("Driver ping timeout."); break
                    except websockets.exceptions.ConnectionClosed: break
                    try:
                        message = json.loads(data)
                        if "image" in message and isinstance(message["image"], str):
                            img_data = message["image"]
                            if len(img_data) > 100:
                                current_image_base64 = img_data
                                await broadcast_to_clients({"image": current_image_base64})
                    except json.JSONDecodeError: pass
                    except Exception as e: logger.error(f"Driver msg process error: {e}")
        except Exception as e: logger.warning(f"Driver connection error: {type(e).__name__}. Retrying...")
        finally:
            if driver_websocket:
                try: await driver_websocket.close()
                except Exception: pass
                driver_websocket = None
            current_image_base64 = None
            await broadcast_to_clients({"status": "driver_disconnected", "image": None})
            await asyncio.sleep(5)

async def broadcast_to_clients(data: Dict):
    if not client_websockets: return
    msg = json.dumps(data)
    disconnected = []
    for ws in client_websockets[:]:
        try: await ws.send_text(msg)
        except Exception: disconnected.append(ws)
    for ws in disconnected:
        if ws in client_websockets: client_websockets.remove(ws)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_websockets.append(websocket)
    try:
        if current_image_base64: await websocket.send_text(json.dumps({"image": current_image_base64}))
        status = "driver_connected" if driver_websocket else "driver_disconnected"
        auto_status = "on" if is_autonomous_running else "off"
        await websocket.send_text(json.dumps({"status": status, "autonomous_status": auto_status}))
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await process_command(message.get("action"), message.get("parameters", {}))
            except json.JSONDecodeError: await websocket.send_text(json.dumps({"error": "Invalid JSON."}))
            except Exception as e: logger.error(f"Client msg error: {e}"); await websocket.send_text(json.dumps({"error": "Server error."}))
    except WebSocketDisconnect: pass
    finally:
        if websocket in client_websockets: client_websockets.remove(websocket)

# --- Command Processing & Autonomous Logic ---
async def process_command(action: Optional[str], parameters: Dict):
    global last_command
    if action in ["describe", "autonomous", "custom"]:
        img_valid = current_image_base64 and is_base64_valid(strip_base64_prefix(current_image_base64))
        if not img_valid:
            msg = f"No valid image for action '{action}'."; logger.warning(msg)
            await broadcast_to_clients({"response": msg, "error": "No valid image."})
            if action == "autonomous" and is_autonomous_running: await stop_autonomous_mode()
            return

    if action == "manual":
        command = parameters.get("command", "stop")
        if command not in VALID_COMMANDS: command = "stop"
        if is_autonomous_running: await stop_autonomous_mode()
        if driver_websocket:
            try:
                await driver_websocket.send(json.dumps({"command": command}))
                last_command = command; resp = f"Executed: {command.replace('_', ' ')}"
                audio = await generate_tts(resp); payload = {"response": resp}
                if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
                await broadcast_to_clients(payload)
            except Exception as e: logger.error(f"Send manual cmd error: {e}"); await broadcast_to_clients({"response": "Send error."})
        else: await broadcast_to_clients({"response": "Robot not connected."})

    elif action == "custom":
        custom_command_text = parameters.get("command", "").strip()
        if not custom_command_text: await broadcast_to_clients({"response": "Custom command empty."}); return
        if is_autonomous_running: await stop_autonomous_mode()
        interpretation_result = await query_ollama_interpret_custom(current_image_base64, custom_command_text)
        final_command = interpretation_result.get("command", "stop")
        if driver_websocket:
            try:
                await driver_websocket.send(json.dumps({"command": final_command}))
                last_command = final_command
                resp = f"Interpreted '{custom_command_text}' as '{final_command}'."
                audio = await generate_tts(resp); payload = {"response": resp}
                if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
                await broadcast_to_clients(payload)
            except Exception as e: logger.error(f"Send custom cmd error: {e}"); await broadcast_to_clients({"response": "Send error."})
        else: await broadcast_to_clients({"response": "Robot not connected."})

    elif action == "describe":
        result = await query_ollama_describe_ui(current_image_base64)
        desc = result.get("description", "Failed description.")
        audio = await generate_tts(desc); payload = {"response": desc}
        if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
        await broadcast_to_clients(payload)

    elif action == "autonomous":
        mode = parameters.get("mode")
        if mode == "on": await start_autonomous_mode()
        elif mode == "off": await stop_autonomous_mode()
        else: await broadcast_to_clients({"error": f"Invalid mode '{mode}'."})

async def start_autonomous_mode():
    global is_autonomous_running, autonomous_task
    if is_autonomous_running: return
    if not driver_websocket: await broadcast_to_clients({"response": "Cannot start: Robot disconnected."}); return
    if not current_image_base64 or not is_base64_valid(strip_base64_prefix(current_image_base64)):
        await broadcast_to_clients({"response": "Cannot start: No valid image."}); return
    is_autonomous_running = True
    if autonomous_task and not autonomous_task.done(): autonomous_task.cancel()
    autonomous_task = asyncio.create_task(autonomous_control())
    logger.info("Autonomous mode ENGAGED.")
    await broadcast_to_clients({"response": "Autonomous mode engaged.", "autonomous_status": "on"})

async def stop_autonomous_mode():
    global is_autonomous_running, autonomous_task, last_command
    if not is_autonomous_running: return
    is_autonomous_running = False
    if autonomous_task and not autonomous_task.done(): autonomous_task.cancel()
    try: await asyncio.wait_for(autonomous_task, timeout=0.5)
    except (asyncio.CancelledError, asyncio.TimeoutError): pass
    autonomous_task = None
    if driver_websocket:
        try: await driver_websocket.send(json.dumps({"command": "stop"}))
        except Exception: pass
    last_command = "stop"
    logger.info("Autonomous mode DISENGAGED.")
    await broadcast_to_clients({"response": "Autonomous mode disengaged.", "autonomous_status": "off"})

async def autonomous_control():
    global last_command
    logger.info("Autonomous control loop STARTED (Road Optimized).")
    while is_autonomous_running:
        if not driver_websocket or not current_image_base64 or not is_base64_valid(strip_base64_prefix(current_image_base64)):
            await stop_autonomous_mode()
            break

        try:
            result = await query_ollama_autonomous(current_image_base64, last_command)
            command = result.get("command", "stop")
            if command != last_command and driver_websocket:
                await driver_websocket.send(json.dumps({"command": command}))
                logger.info(f"Auto command: {command}")
                tts_text = f"Driving {command.replace('_', ' ')}"
                audio_b64 = await generate_tts(tts_text)
                payload = {"autonomous_command": command, "response": tts_text}
                if audio_b64: payload["audio"] = f"data:audio/mp3;base64,{audio_b64}"
                await broadcast_to_clients(payload)
                last_command = command
        except Exception as e:
            logger.error(f"Auto control error: {e}")
            if driver_websocket:
                await driver_websocket.send(json.dumps({"command": "stop"}))
            last_command = "stop"
            await stop_autonomous_mode()
            break

        await asyncio.sleep(AUTONOMOUS_INTERVAL)  # 실시간 반응성 조정

    logger.info("Autonomous control loop STOPPED.")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI - JetBot IP: {DRIVER_WEBSOCKET_URL}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
