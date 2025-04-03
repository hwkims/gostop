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
MODEL_NAME = "granite3.2-vision"
DRIVER_WEBSOCKET_URL = "ws://192.168.137.181:8766" # !! VERIFY JETBOT IP !!
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

OLLAMA_OPTIONS = {"temperature": 0.0} # Consistent decisions

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
    yield # App runs
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
app = FastAPI(title="JetBot Vision Control v19 (Simplified Prompts)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Helper Functions ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists(): return HTMLResponse("Error: index.html not found", 404)
    try:
        with open(index_path, "r", encoding="utf-8") as f: return HTMLResponse(f.read())
    except Exception as e: logger.error(f"Read index.html error: {e}"); return HTMLResponse("Server error", 500)

async def generate_tts(text: str) -> Optional[str]:
    """Generates TTS audio and returns base64 string."""
    # TTS is generated here when called by process_command or autonomous_control
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

# --- Ollama Interaction (Two-Stage, Simplified Prompts) ---

async def query_ollama_analyze_scene(image_data: str) -> Optional[Dict[str, Any]]:
    """Stage 1: Analyze image, extract structured facts (No examples)."""
    cleaned_data = strip_base64_prefix(image_data)
    if not is_base64_valid(cleaned_data):
        logger.error("AnalyzeScene: Invalid image data.")
        return None

    # Simplified prompt, no example
    prompt_text = (
        "Analyze the robot's camera image. Extract the following details ONLY. "
        "Use 'unknown' if uncertain or not visible. Distances in meters."
        # "\n- immediate_obstacle_within_1m: (true/false)"
        # "\n- stopped_vehicle_within_3m: (true/false)"
        # "\n- traffic_light_state: ('red'/'yellow'/'green'/'off'/'unknown')"
        # "\n- path_clearance: ('clear_long' [>4m] / 'clear_medium' [2-4m] / 'clear_short' [0.8-2m] / 'blocked' [<0.8m] / 'unknown')"
        # "\n- path_direction: ('straight' / 'gentle_left' / 'moderate_left' / 'sharp_left' / 'gentle_right' / 'moderate_right' / 'sharp_right' / 'unknown')"
        "\nRespond ONLY with a valid JSON object containing these keys."

    )

    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [cleaned_data], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    # logger.info("Sending scene analysis request to Ollama...")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            # logger.debug(f"Ollama Analysis Raw Response: {inner_json_str}")
            if not inner_json_str: logger.warning("Ollama analysis returned empty."); return None
            try:
                analysis_result = json.loads(inner_json_str)
                required_keys = ["immediate_obstacle_within_1m", "stopped_vehicle_within_3m", "traffic_light_state", "path_clearance", "path_direction"]
                if not all(key in analysis_result for key in required_keys):
                     logger.warning(f"Analysis result missing keys: {analysis_result}")
                     return None
                # logger.info(f"Scene Analysis Result: {analysis_result}")
                return analysis_result
            except json.JSONDecodeError: logger.error(f"Failed decode analysis JSON: '{inner_json_str[:150]}...'"); return None
    except Exception as e: logger.error(f"Ollama analysis query failed: {e}", exc_info=True); return None

async def query_ollama_decide_command(analysis: Dict[str, Any], last_cmd: str) -> Dict[str, str]:
    """Stage 2: Decide command based on analysis text (Natural Language Instructions)."""

    analysis_text = ( # Format analysis for the model
        f"Current Scene Analysis:\n"
        f"- Obstacle < 1m: {analysis.get('immediate_obstacle_within_1m', 'unknown')}\n"
        f"- Stopped Vehicle < 3m: {analysis.get('stopped_vehicle_within_3m', 'unknown')}\n"
        f"- Traffic Light: {analysis.get('traffic_light_state', 'unknown')}\n"
        f"- Path Clearance: {analysis.get('path_clearance', 'unknown')}\n"
        f"- Path Direction: {analysis.get('path_direction', 'unknown')}\n"
        f"- Last Command: {last_cmd}"
    )

    valid_cmds_str = ", ".join(VALID_COMMANDS)

    # More natural language, less prescriptive prompt
    prompt_text = (
        f"**You are the decision module for a robot driver.** You are given the 'Current Scene Analysis' below. Your task is to choose the single safest and most appropriate command from the list: {valid_cmds_str}."
        f"\n\n{analysis_text}"
        "\n\n**Driving Priorities:**"
        "\n1.  **Safety First:** Immediately stop ('stop') if the analysis indicates critical hazards like obstacles under 1m, stopped vehicles under 3m, red lights, or a blocked path."
        "\n2.  **Caution:** If the analysis shows a yellow light or potential hazards needing slower speed or slight avoidance, choose a cautious command like 'forward_slow' or 'left/right_slow'."
        "\n3.  **Progress:** If safety and caution rules don't apply, select a command to make smooth progress. Match the command's speed and direction ('forward/left/right' combined with 'slow/medium/fast') to the analyzed 'Path Clearance' and 'Path Direction'. Drive faster on clear, straight paths; slower on shorter or curved paths."
        "\n4.  **Uncertainty:** If the analysis is 'unknown' or unclear, default to 'stop'."
        "\n\nChoose the best command based on these priorities and the provided analysis."
        " Respond **ONLY** with the chosen command in strict JSON format: {\"command\": \"<chosen_command>\"}" # Only command
    )

    data = {"model": MODEL_NAME, "prompt": prompt_text, "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    # logger.info(f"Sending command decision request...")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data) # No image sent
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            # logger.debug(f"Ollama Decision Raw Response: {inner_json_str}")
            if not inner_json_str: logger.warning("Ollama decision returned empty. Stopping."); return {"command": "stop"}
            try:
                result = json.loads(inner_json_str)
                command = result.get("command")
                if command not in VALID_COMMANDS:
                    logger.warning(f"Ollama invalid decision cmd: '{command}'. Stopping."); command = "stop"
                return {"command": command}
            except json.JSONDecodeError:
                logger.error(f"Failed decode decision JSON: '{inner_json_str[:100]}...'. Stopping."); return {"command": "stop"}
    except Exception as e:
        logger.error(f"Ollama decision query failed: {e}", exc_info=True)
        return {"command": "stop"}


async def query_ollama_describe_ui(image_data: str) -> Dict:
    """Requests simple description for UI display (independent of autonomous)."""
    cleaned_data = strip_base64_prefix(image_data)
    if not is_base64_valid(cleaned_data): return {"description": "Error: Invalid image data."}
    # Simplified describe prompt for UI
    prompt_text = (
        "Observe via robot camera. Concisely describe: "
        "Path status? Nearest obstacle (type, dist)? "
        "Scene type & visible traffic lights? "
        "Strict JSON: {\"description\": \"<summary>\"}"
    )
    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [cleaned_data], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            if not inner_json_str: return {"description": "Empty response."}
            try:
                result = json.loads(inner_json_str)
                desc_val=result.get("description"); desc = str(desc_val).strip() if isinstance(desc_val, str) else str(desc_val) if desc_val else ""
                return {"description": desc or "Empty description."}
            except json.JSONDecodeError: return {"description": f"Format error. Response: {inner_json_str}"}
    except Exception as e: logger.error(f"Ollama UI describe query error: {e}"); return {"description": "Error during description."}

# --- WebSocket Handling (Driver & Clients) ---
async def connect_to_driver():
    # ... (No changes needed from v17) ...
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
                    except Exception as e: logger.error(f"Driver msg process error: {e}", exc_info=True)
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
    # ... (No changes needed from v17) ...
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
    # ... (No changes needed from v17) ...
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
    except Exception as e: logger.error(f"Client WS error: {e}", exc_info=True)
    finally:
        if websocket in client_websockets: client_websockets.remove(websocket)

# --- Command Processing & Autonomous Logic ---
async def process_command(action: Optional[str], parameters: Dict, sender_ws: Optional[WebSocket] = None):
    # ... (No significant changes needed from v17 for custom/manual/describe actions) ...
    global last_command
    if action in ["describe", "autonomous"]:
        img_valid = current_image_base64 and is_base64_valid(strip_base64_prefix(current_image_base64))
        if not img_valid:
            msg = "No valid image available."; logger.warning(f"Action '{action}' failed: {msg.lower()}")
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
                audio = await generate_tts(resp); payload = {"response": resp};
                if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
                await broadcast_to_clients(payload)
            except Exception as e: logger.error(f"Send manual cmd '{command}' error: {e}"); await broadcast_to_clients({"response": f"Send error."})
        else: await broadcast_to_clients({"response": "Robot not connected."})

    elif action == "custom":
        custom_command_text = parameters.get("command", "").strip()
        if not custom_command_text: custom_command_text = "stop"
        if is_autonomous_running: await stop_autonomous_mode()
        if driver_websocket:
            try:
                await driver_websocket.send(json.dumps({"command": custom_command_text}))
                logger.info(f"Sent custom command: '{custom_command_text}'")
                resp = f"Sent custom: '{custom_command_text}'"
                audio = await generate_tts(resp); payload = {"response": resp};
                if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
                await broadcast_to_clients(payload)
            except Exception as e: logger.error(f"Send custom cmd '{custom_command_text}' error: {e}"); await broadcast_to_clients({"response": f"Send error."})
        else: await broadcast_to_clients({"response": "Robot not connected."})

    elif action == "describe":
        # Use the dedicated UI description function
        result = await query_ollama_describe_ui(current_image_base64)
        desc = result.get("description", "Failed description.")
        audio = await generate_tts(desc); payload = {"response": desc};
        if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
        await broadcast_to_clients(payload)

    elif action == "autonomous":
        mode = parameters.get("mode")
        if mode == "on": await start_autonomous_mode()
        elif mode == "off": await stop_autonomous_mode()
        else: await broadcast_to_clients({"error": f"Invalid mode '{mode}'."})


async def start_autonomous_mode():
    # ... (No changes needed from v17) ...
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
    # ... (No changes needed from v17) ...
    global is_autonomous_running, autonomous_task, last_command
    if not is_autonomous_running: return
    is_autonomous_running = False
    if autonomous_task:
        if not autonomous_task.done(): autonomous_task.cancel()
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
    """Autonomous loop: Analyze scene, Decide command, Send, Repeat reactively."""
    global last_command
    logger.info("Autonomous control loop STARTED (Two-Stage, Natural Prompts).")
    while is_autonomous_running:
        img_to_process = current_image_base64

        if img_to_process and is_base64_valid(strip_base64_prefix(img_to_process)) and driver_websocket:
            command = "stop" # Default to stop for safety
            try:
                # === STAGE 1: Analyze Scene ===
                analysis_result = await query_ollama_analyze_scene(img_to_process)

                if analysis_result:
                    # === STAGE 2: Decide Command ===
                    decision_result = await query_ollama_decide_command(analysis_result, last_command)
                    command = decision_result.get("command", "stop") # Use decided command
                else:
                    logger.warning("Scene analysis failed, defaulting to stop.")
                    # command remains "stop"

                # === Send Command ===
                if is_autonomous_running and driver_websocket:
                    await driver_websocket.send(json.dumps({"command": command}))

                    # Announce command via TTS if it changed
                    if command != last_command:
                        logger.info(f"Auto command: {command}") # Log final command
                        tts_text = command.replace('_', ' ').capitalize()
                        audio_b64 = await generate_tts(tts_text)
                        payload = {"autonomous_command": command}
                        if audio_b64: payload["audio"] = f"data:audio/mp3;base64,{audio_b64}"
                        await broadcast_to_clients(payload)

                    last_command = command # Update state

            except Exception as e:
                logger.error(f"Error in auto control cycle: {e}", exc_info=True)
                if is_autonomous_running and driver_websocket: # Safety stop
                    try: await driver_websocket.send(json.dumps({"command": "stop"}))
                    except Exception: pass
                last_command = "stop"
                await asyncio.sleep(0.5) # Pause after error
        # Handle missing prerequisites
        elif not driver_websocket: await asyncio.sleep(0.5)
        elif not img_to_process or not is_base64_valid(strip_base64_prefix(img_to_process)): await asyncio.sleep(0.1)
        else: break # Exit loop if state changed

        await asyncio.sleep(0.05) # Yield control briefly

    logger.info("Autonomous control loop STOPPED.")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # Optional: Increase log level for httpx if Ollama logs are too noisy
    # logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info(f"Starting FastAPI - Check JetBot IP: {DRIVER_WEBSOCKET_URL}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
