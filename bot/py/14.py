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
app = FastAPI(title="JetBot Vision Control v22 (General Description)", lifespan=lifespan)
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
    """Requests general description of the robot's view for UI."""
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string): return {"description": "Error: Invalid image data."}

    # --- General Describe Prompt ---
    prompt_text = (
        "You are observing through a robot's forward camera. "
        "**Describe what you see in the image factually and concisely.** Focus on the main elements relevant to the robot's environment and potential movement."
        "**DO NOT comment on the image quality (e.g., blurry, small). Describe ONLY the visible content.**"
        " Respond **strictly** JSON: {\"description\": \"<factual scene summary>\"}"
    )
    # --- End Describe Prompt ---

    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            if not inner_json_str: return {"description": "Empty response."}
            try:
                result = json.loads(inner_json_str)
                desc_val = result.get("description"); desc = str(desc_val).strip() if isinstance(desc_val, str) else str(desc_val) if desc_val else ""
                # Check if model still complained despite instructions
                if "small" in desc.lower() or "blurry" in desc.lower() or "clarity" in desc.lower() or "quality" in desc.lower():
                     logger.warning(f"Model description *still* mentioned quality despite prompt: '{desc[:150]}...'")
                     # Try to return something useful anyway, maybe filter the quality part?
                     # For now, just return it as is.
                return {"description": desc or "Empty description."}
            except json.JSONDecodeError: logger.error(f"Failed decode describe JSON: '{inner_json_str[:100]}...'"); return {"description": f"Format error. Response: {inner_json_str}"}
    except Exception as e: logger.error(f"Ollama describe query error: {e}"); return {"description": "Error during description."}

async def query_ollama_autonomous(image_data: str, last_cmd: str = "stop") -> Dict[str, str]:
    """Single-stage: Analyze image and decide command directly."""
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string):
        logger.error("Invalid image for autonomous decision. Stopping.")
        return {"command": "stop"}

    valid_cmds_str = ", ".join(VALID_COMMANDS)

    # --- Autonomous Prompt (Single Stage, Natural Language) ---
    prompt_text = (
        f"**Act as an expert AI driver for a robot.** Analyze the provided camera image showing the robot's forward view. "
        f"Your goal is **absolute safety first**, then smooth, efficient progress. Last command was '{last_cmd}'."
        f" Choose the **single best command** from the list: {valid_cmds_str} based on your direct visual analysis."
        "\n\n**Driving Priorities (Apply based on image):**"
        "\n1.  **Critical Safety Stop:** Immediately command 'stop' if you see critical hazards: RED lights, stopped vehicles/obstacles blocking path (<3m), ANY obstacle very close (<0.8m), or impassable conditions."
        "\n2.  **Caution:** If you see YELLOW lights or obstacles requiring careful maneuvers (0.8m-3m away), use 'forward_slow' or 'left/right_slow'."
        "\n3.  **Proceed Safely:** If no immediate safety/caution triggers apply, analyze the path's geometry and clearance. Choose an appropriate 'forward/left/right' command with 'slow/medium/fast' speed to match the visible conditions (faster for long clear straight paths, slower for shorter/curved paths)."
        "\n4.  **Uncertainty:** If unsure about the scene or safe action, default to 'stop'."
        "\n\nSynthesize your visual analysis and these priorities to select the one best command."
        " Respond **ONLY** with the chosen command in strict JSON format: {\"command\": \"<chosen_command_from_list>\"}"
    )
    # --- End Prompt ---

    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    # logger.info(f"Sending single-stage autonomous request...")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            # logger.debug(f"Ollama Auto Raw Response: {inner_json_str}")
            if not inner_json_str: logger.warning("Ollama empty auto response. Stopping."); return {"command": "stop"}
            try:
                result = json.loads(inner_json_str)
                command = result.get("command")
                if command not in VALID_COMMANDS:
                    logger.warning(f"Ollama invalid cmd: '{command}'. Stopping."); command = "stop"
                return {"command": command}
            except json.JSONDecodeError:
                logger.error(f"Failed decode auto JSON: '{inner_json_str[:100]}...'. Stopping."); return {"command": "stop"}
    except Exception as e:
        logger.error(f"Ollama auto query failed: {e}", exc_info=True)
        return {"command": "stop"}

async def query_ollama_custom_response(image_data: str, custom_prompt: str) -> Dict[str, str]:
    """Interprets a user's custom command text in the context of the image and responds with text."""
    base64_string = strip_base64_prefix(image_data)
    if not is_base64_valid(base64_string):
        logger.error("Invalid image for custom command response.")
        return {"response": "Error: Cannot process custom command without a valid image."}

    # --- Custom Command Text Response Prompt ---
    prompt_text = (
        f"**You are a helpful robot assistant.** A user looking at the robot's camera view (provided in the image) gave the instruction: '{custom_prompt}'. "
        f"**Analyze the current camera image** in the context of this instruction. "
        f"Provide a **brief, factual text response** describing what you see relevant to the user's instruction OR explain why it cannot be done safely based on the visual evidence. "
        f"For example, if asked 'check left' and a wall is visible, respond 'There is a wall to the left.' If asked 'go through door' and door is closed, respond 'The door ahead appears closed.' If asked for something unrelated, state that."
        f" Do not suggest robot commands."
        f" Respond **strictly** in JSON format: {{\"response\": \"<your textual response>\"}}"
    )
    # --- End Prompt ---

    data = {"model": MODEL_NAME, "prompt": prompt_text, "images": [base64_string], "stream": False, "format": "json", "options": OLLAMA_OPTIONS}
    logger.info(f"Sending custom command text interpretation request for: '{custom_prompt}'")

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=data)
            response.raise_for_status()
            response_json = response.json(); inner_json_str = response_json.get("response")
            logger.debug(f"Ollama Custom Text Response Raw: {inner_json_str}")
            if not inner_json_str: logger.warning("Ollama empty custom response."); return {"response": "Model gave no response."}
            try:
                result = json.loads(inner_json_str)
                response_text = result.get("response", "No response text found.")
                logger.info(f"Custom command response: '{response_text}'")
                return {"response": response_text}
            except json.JSONDecodeError:
                logger.error(f"Failed decode custom JSON response: '{inner_json_str[:100]}...'"); return {"response": f"Format error: {inner_json_str}"}
    except Exception as e:
        logger.error(f"Ollama custom query for response failed: {e}", exc_info=True)
        return {"response": "Error interpreting custom command."}


# --- WebSocket Handling (Driver & Clients) ---
async def connect_to_driver():
    # ... (No changes needed) ...
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
    # ... (No changes needed) ...
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
    # ... (No changes needed) ...
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
    global last_command
    # Image check required for describe, autonomous, and custom response
    if action in ["describe", "autonomous", "custom"]:
        img_valid = current_image_base64 and is_base64_valid(strip_base64_prefix(current_image_base64))
        if not img_valid:
            msg = f"No valid image available for action '{action}'."; logger.warning(msg)
            await broadcast_to_clients({"response": msg, "error": "No valid image."})
            # Only stop autonomous mode if it was the action requested without image
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

    elif action == "custom": # *** MODIFIED: Uses AI for TEXT response, NO robot command ***
        custom_command_text = parameters.get("command", "").strip()
        if not custom_command_text:
             await broadcast_to_clients({"response": "Custom command empty.", "error": "Empty input."}); return

        # We don't necessarily stop autonomous for a custom query
        # if is_autonomous_running: await stop_autonomous_mode() # Decide if custom query should interrupt auto

        # Get TEXT response from Ollama based on image and custom text
        logger.info(f"Getting AI response for custom command: '{custom_command_text}'")
        interpretation_result = await query_ollama_custom_response(current_image_base64, custom_command_text)
        response_text = interpretation_result.get("response", "Failed to interpret custom command.")

        logger.info(f"Custom command response: '{response_text}'")

        # Send the TEXT response (and TTS) to the UI
        audio = await generate_tts(response_text); payload = {"response": response_text};
        if audio: payload["audio"] = f"data:audio/mp3;base64,{audio}"
        await broadcast_to_clients(payload)
        # NOTE: No command is sent to the driver for "custom" action anymore.


    elif action == "describe":
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
    # ... (No changes needed) ...
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
    # ... (No changes needed) ...
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
    """Autonomous loop: Single-stage Ollama call, send command reactively."""
    global last_command
    logger.info("Autonomous control loop STARTED (Single Stage, NL Prompt).")
    while is_autonomous_running:
        img_to_process = current_image_base64

        if img_to_process and is_base64_valid(strip_base64_prefix(img_to_process)) and driver_websocket:
            command = "stop" # Default safety
            try:
                result = await query_ollama_autonomous(img_to_process, last_command)
                command = result.get("command", "stop")

                if is_autonomous_running and driver_websocket:
                    await driver_websocket.send(json.dumps({"command": command}))
                    if command != last_command:
                        logger.info(f"Auto command: {command}")
                        tts_text = command.replace('_', ' ').capitalize()
                        audio_b64 = await generate_tts(tts_text)
                        payload = {"autonomous_command": command}
                        if audio_b64: payload["audio"] = f"data:audio/mp3;base64,{audio_b64}"
                        await broadcast_to_clients(payload)
                    last_command = command
            except Exception as e:
                logger.error(f"Error in auto control cycle: {e}", exc_info=True)
                if is_autonomous_running and driver_websocket: # Safety stop
                    try: await driver_websocket.send(json.dumps({"command": "stop"}))
                    except Exception: pass
                last_command = "stop"
                await asyncio.sleep(0.5)
        elif not driver_websocket: await asyncio.sleep(0.5)
        elif not img_to_process or not is_base64_valid(strip_base64_prefix(img_to_process)): await asyncio.sleep(0.1)
        else: break

        await asyncio.sleep(0.05) # Yield briefly

    logger.info("Autonomous control loop STOPPED.")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting FastAPI - Check JetBot IP: {DRIVER_WEBSOCKET_URL}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
