# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import asyncio
import struct
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import tempfile
import time
import uuid
import yaml  # For loading presets
import numpy as np
import librosa  # For potential direct use if needed, though utils.py handles most
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal
import webbrowser  # For automatic browser opening
import threading  # For automatic browser opening

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_ssl_config,
    get_log_file_path,
    get_output_path,
    get_reference_audio_path,
    get_predefined_voices_path,
    get_ui_title,
    get_gen_default_temperature,
    get_gen_default_exaggeration,
    get_gen_default_cfg_weight,
    get_gen_default_seed,
    get_gen_default_speed_factor,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine interface
from models import (  # Pydantic models
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input")
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"  # Add "mp3"
    speed: float = 1.0
    seed: Optional[int] = None
    language: Optional[str] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# --- User Activity Logging Switch ---
# 'server.enable_logging' (default off) decides whether anything is recorded once
# the server is up and serving. The rule is deliberately blunt: everything logged
# while serving is, one way or another, a trace of what someone did with the app -
# the text they synthesized, the files they uploaded, the voices they chose. So
# once startup finishes, records are dropped unless logging is switched on.
#
# Startup and shutdown are always logged: they describe the server's own
# lifecycle, contain nothing a user did, and are what makes a failed boot
# diagnosable even with logging off.
_server_is_serving = False
_logging_enabled = config_manager.get_bool("server.enable_logging", False)


class UserActivityFilter(logging.Filter):
    """Suppresses log records emitted while serving, unless logging is enabled."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _logging_enabled or not _server_is_serving


_user_activity_filter = UserActivityFilter()


def apply_logging_state() -> None:
    """
    Refreshes the cached enable_logging flag and makes sure every live handler
    carries the filter.

    Handler-level filtering is the only reliable interception point: a record
    propagating up from a child logger skips ancestor *logger* filters, but must
    pass the filters on every handler that emits it. Uvicorn installs its own
    handlers after this module is imported, hence re-running this at startup.
    """
    global _logging_enabled
    _logging_enabled = config_manager.get_bool("server.enable_logging", False)

    logger_names = ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]
    for target in [logging.getLogger()] + [logging.getLogger(n) for n in logger_names]:
        for handler in target.handlers:
            if not any(isinstance(f, UserActivityFilter) for f in handler.filters):
                handler.addFilter(_user_activity_filter)

    # One line per request with method, path and client address - pure user
    # activity, and it bypasses the app's own loggers entirely.
    logging.getLogger("uvicorn.access").disabled = not _logging_enabled


apply_logging_state()

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()  # For coordinating browser opening


def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    global _server_is_serving
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            get_reference_audio_path(),
            get_predefined_voices_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")
            host_address = get_host()
            server_port = get_port()
            browser_thread = threading.Thread(
                target=lambda: _delayed_browser_open(host_address, server_port),
                daemon=True,
            )
            browser_thread.start()

        # Re-run now that uvicorn has installed its own handlers, so they carry
        # the filter too.
        apply_logging_state()
        if not _logging_enabled:
            logger.info(
                "Activity logging is off ('server.enable_logging'). Nothing done in "
                "the app will be recorded until it is switched on."
            )

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        _server_is_serving = True
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        _server_is_serving = True
        yield
    finally:
        # Shutdown is lifecycle, not user activity, so let it through.
        _server_is_serving = False
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")


# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="2.0.2",  # Version Bump
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Static Files and HTML Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )


@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")


outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

templates = Jinja2Templates(directory=str(ui_static_path))

# --- API Endpoints ---

# --- Audio Stitching Helper Functions ---
# These functions support smart audio chunk concatenation with crossfading


def _generate_equal_power_curves(n_samples: int):
    """
    Generate equal-power crossfade curves using cos²/sin² functions.
    These curves maintain perceptually constant loudness during transitions.

    Args:
        n_samples: Number of samples in the fade region

    Returns:
        Tuple of (fade_out, fade_in) numpy arrays
    """
    t = np.linspace(0, np.pi / 2, n_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2  # 1 → 0
    fade_in = np.sin(t) ** 2  # 0 → 1
    return fade_out, fade_in


def _crossfade_with_overlap(
    chunk_a: np.ndarray, chunk_b: np.ndarray, fade_samples: int
) -> np.ndarray:
    """
    Perform true crossfade by overlapping and summing audio regions.

    This creates a seamless transition by:
    1. Taking the tail of chunk_a and head of chunk_b
    2. Applying equal-power fade curves
    3. Summing the overlapped regions

    Result length = len(chunk_a) + len(chunk_b) - fade_samples

    Args:
        chunk_a: First audio chunk (numpy float32 array)
        chunk_b: Second audio chunk (numpy float32 array)
        fade_samples: Number of samples to overlap

    Returns:
        Crossfaded audio as numpy float32 array
    """
    # Handle edge cases
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples <= 0:
        return np.concatenate([chunk_a, chunk_b])

    fade_out, fade_in = _generate_equal_power_curves(fade_samples)

    # Extract overlap regions
    a_tail = chunk_a[-fade_samples:]
    b_head = chunk_b[:fade_samples]

    # Crossfade: weighted sum of overlapping regions
    crossfaded_region = (a_tail * fade_out) + (b_head * fade_in)

    # Assemble: [chunk_a without tail] + [crossfaded region] + [chunk_b without head]
    return np.concatenate(
        [chunk_a[:-fade_samples], crossfaded_region, chunk_b[fade_samples:]]
    )


def _apply_edge_fades(
    chunk: np.ndarray, fade_samples: int, fade_in: bool = True, fade_out: bool = True
) -> np.ndarray:
    """
    Apply minimal linear edge fades for click protection.

    This is used in fallback mode when full crossfading is disabled.
    Linear fades are acceptable for ultra-short safety fades (2-3ms).

    Args:
        chunk: Audio chunk (numpy array)
        fade_samples: Number of samples to fade
        fade_in: Whether to apply fade-in at start
        fade_out: Whether to apply fade-out at end

    Returns:
        Audio chunk with edge fades applied (numpy float32 array)
    """
    # Skip if chunk is too short for fading
    if len(chunk) < fade_samples * 2:
        return chunk.astype(np.float32, copy=False)

    result = chunk.astype(np.float32, copy=True)

    if fade_in:
        result[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    if fade_out:
        result[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

    return result


def _remove_dc_offset(
    audio: np.ndarray, sample_rate: int, cutoff_hz: float = 15.0
) -> np.ndarray:
    """
    Remove DC offset using a high-pass Butterworth filter.

    DC offset can cause low-frequency thumps when concatenating audio chunks.
    This applies a 2nd-order high-pass filter at the specified cutoff frequency.

    Args:
        audio: Audio data (numpy array)
        sample_rate: Sample rate in Hz
        cutoff_hz: High-pass filter cutoff frequency (default 15 Hz)

    Returns:
        Audio with DC offset removed (numpy float32 array)

    Note:
        Requires scipy. If scipy is not available, returns audio unchanged
        with a warning logged.
    """
    try:
        from scipy.signal import butter, filtfilt

        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist

        # 2nd-order Butterworth high-pass filter
        b, a = butter(2, normalized_cutoff, btype="high")

        # Zero-phase filtering (no phase distortion)
        return filtfilt(b, a, audio).astype(np.float32)

    except ImportError:
        logger.warning(
            "scipy not available for DC offset removal. "
            "Install scipy to enable this feature: pip install scipy"
        )
        return audio.astype(np.float32, copy=False)
    except Exception as e:
        logger.error(f"DC offset removal failed: {e}")
        return audio.astype(np.float32, copy=False)


def _create_wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Build a WAV header with 0xFFFFFFFF data size for streaming (size unknown upfront)."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    header = struct.pack("<4sI4s", b"RIFF", 0xFFFFFFFF, b"WAVE")
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
    )
    data_header = struct.pack("<4sI", b"data", 0xFFFFFFFF)
    return header + fmt_chunk + data_header


def _float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    """Convert a float32 numpy array in [-1, 1] to int16 PCM bytes."""
    clipped = np.clip(audio_np, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


# --- End Audio Stitching Helper Functions ---


# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Model Information ---
@app.get("/api/model-info", tags=["Model Information"])
async def get_model_info_endpoint():
    """
    Returns detailed information about the currently loaded TTS model.
    This endpoint is used by the UI to display model status and
    conditionally show features like paralinguistic tags.
    """
    logger.debug("Request received for /api/model-info")
    try:
        model_info = engine.get_model_info()
        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data():
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, presets, and model information.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()
        reference_files = utils.get_valid_reference_files()
        predefined_voices = utils.get_predefined_voices()

        # Get model information for UI
        model_info = engine.get_model_info()

        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoiceMode": None,
            "submittedPredefinedVoice": None,
            "submittedCloneFile": None,
        }

        return {
            "config": full_config,
            "reference_files": reference_files,
            "predefined_voices": predefined_voices,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,  # NEW: Include model information
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )


# --- Configuration Management API Endpoints ---
# Sections whose settings are only read at startup.
RESTART_REQUIRED_SECTIONS = ("server", "tts_engine", "paths", "model")
# Exceptions within those sections that are applied live, so saving them alone
# must not tell the user to restart. Dotted paths from the config root.
LIVE_APPLIED_SETTINGS = frozenset({"server.enable_logging"})


def _settings_need_restart(partial_update: Dict[str, Any], prefix: str = "") -> bool:
    """Reports whether a config update touches anything only read at startup."""
    for key, value in partial_update.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            if _settings_need_restart(value, f"{path}."):
                return True
        elif (
            path.split(".")[0] in RESTART_REQUIRED_SECTIONS
            and path not in LIVE_APPLIED_SETTINGS
        ):
            return True
    return False


@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(request: Request):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            # Takes effect immediately - a user turning logging off should not have
            # to restart before it stops recording, nor lose the next few minutes
            # of diagnostics after turning it on.
            apply_logging_state()

            restart_needed = _settings_need_restart(partial_update)
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@app.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint():
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint():
    """
    Triggers a hot-swap of the TTS model engine.
    Unloads the current model, clears VRAM, and loads the model defined in config.
    """
    logger.info("Request received for /restart_server (Model Hot-Swap).")

    try:
        # Attempt to reload the engine with the new configuration
        success = engine.reload_model()

        if success:
            model_info = engine.get_model_info()
            new_model_name = model_info.get("class_name", "Unknown Model")
            new_model_type = model_info.get("type", "unknown")
            message = f"Model hot-swap successful. Now running: {new_model_name} ({new_model_type})"
            logger.info(message)

            # restart_needed=False because we just performed the hot-swap successfully
            return UpdateStatusResponse(message=message, restart_needed=False)
        else:
            error_msg = "Model reload failed. The server may be in an inconsistent state. Check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during model hot-swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model reload: {str(e)}",
        )


@app.post("/api/unload", tags=["Configuration"])
async def unload_model_endpoint():
    """
    Unloads the TTS model and releases all CUDA/GPU memory.
    The model will need to be reloaded (via /restart_server) before TTS requests can be processed.
    """
    logger.info("Request received for /api/unload (Model Unload).")

    try:
        success = engine.unload_model()

        if success:
            logger.info("Model successfully unloaded and GPU memory released.")
            return {"status": "unloaded"}
        else:
            error_msg = "Model unload failed. Check logs for details."
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error during model unload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during model unload: {str(e)}",
        )


# --- UI Helper API Endpoints ---
@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """Returns a list of valid reference audio filenames (.wav, .mp3)."""
    logger.debug("Request for /get_reference_files.")
    try:
        return utils.get_valid_reference_files()
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )


@app.get("/preview_reference", tags=["UI Helpers"])
async def preview_reference_audio_endpoint(
    filename: str,
    pitch: float = 0.0,
    speed: float = 1.0,
):
    """
    Returns the reference audio exactly as the model would receive it: capped to
    the maximum duration and with the requested pitch/speed applied.

    Backs the UI's "Play Sample" button. Because it goes through the same
    preparation and cache as generation, previewing a setting also warms the file
    that the next generation will reuse.
    """
    logger.debug(f"Preview request for '{filename}' (pitch={pitch}, speed={speed}).")

    if not -12.0 <= pitch <= 12.0:
        raise HTTPException(
            status_code=400, detail="Pitch must be between -12 and +12 semitones."
        )
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0.")

    ref_dir = get_reference_audio_path(ensure_absolute=True)
    try:
        source_path = utils.safe_resolve_within(ref_dir, filename)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid reference audio filename.")

    if not source_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Reference audio '{filename}' not found."
        )

    max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
    prepared_path = utils.prepare_reference_audio(
        source_path, max_dur, pitch_semitones=pitch, speed_factor=speed
    )
    if prepared_path is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not prepare a preview of '{filename}'.",
        )

    return FileResponse(
        path=str(prepared_path),
        media_type="audio/wav",
        filename=f"preview_{source_path.stem}.wav",
        # The cache key already encodes the file and its settings, but the browser
        # cannot know that, so keep it from replaying a stale preview.
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"]
)
async def get_predefined_voices_api():
    """Returns a list of predefined voices with display names and filenames."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        return utils.get_predefined_voices()
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )


# --- File Upload Endpoints ---
@app.post("/upload_reference", tags=["File Management"])
async def upload_reference_audio_endpoint(
    files: List[UploadFile] = File(...), denoise: bool = Form(False)
):
    """
    Handles uploading of reference audio files (.wav, .mp3) for voice cloning.
    Validates files and saves them to the configured reference audio path.

    Clips longer than 'audio_output.max_reference_duration_sec' are trimmed to
    that length rather than rejected, since the model only conditions on the
    first part of a reference anyway.

    Clips shorter than 'audio_output.min_reference_duration_sec' are unusable on
    their own for cloning, so when two or more of them arrive in the same upload
    they are chained into a single reference file, separated by
    'audio_output.reference_combine_gap_ms' of silence. Files that already meet
    the minimum are saved individually, exactly as before.

    When 'denoise' is set, background noise is stripped from each clip with
    DeepFilterNet before any chaining happens, so every clip is cleaned against
    its own noise profile.

    Uploads are staged in a temporary directory first, so a batch that gets
    combined leaves only the combined file behind rather than its parts.
    """
    logger.info(
        f"Request to /upload_reference with {len(files)} file(s). denoise={denoise}"
    )
    ref_path = get_reference_audio_path(ensure_absolute=True)
    max_duration = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
    min_duration = config_manager.get_float(
        "audio_output.min_reference_duration_sec", 5.0
    )
    gap_ms = config_manager.get_int("audio_output.reference_combine_gap_ms", 3000)

    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []
    upload_warnings: List[Dict[str, str]] = []
    combined_filename: Optional[str] = None
    combined_sources: List[str] = []
    denoised_any = False
    denoise_failure_reported = False

    # (safe_filename, staged_path, duration_sec_or_None)
    staged_files: List[tuple] = []

    with tempfile.TemporaryDirectory(prefix="ref_upload_") as staging_dir_str:
        staging_dir = Path(staging_dir_str)

        for file in files:
            if not file.filename:
                upload_errors.append(
                    {"filename": "Unknown", "error": "File received with no filename."}
                )
                logger.warning("Upload attempt with no filename.")
                continue

            safe_filename = utils.sanitize_filename(file.filename)

            try:
                if not (
                    safe_filename.lower().endswith(".wav")
                    or safe_filename.lower().endswith(".mp3")
                ):
                    raise ValueError(
                        "Invalid file type. Only .wav and .mp3 are allowed."
                    )

                if (ref_path / safe_filename).exists():
                    logger.info(
                        f"Reference file '{safe_filename}' already exists. Skipping duplicate upload."
                    )
                    if safe_filename not in uploaded_filenames_successfully:
                        uploaded_filenames_successfully.append(safe_filename)
                    continue

                staged_path = utils.get_unique_destination(staging_dir, safe_filename)
                with open(staged_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Duration is deliberately not checked here: an over-long clip is
                # trimmed below rather than rejected.
                is_valid, validation_msg = utils.validate_reference_audio(staged_path)
                if not is_valid:
                    logger.warning(
                        f"Uploaded file '{safe_filename}' failed validation: {validation_msg}. Discarding."
                    )
                    upload_errors.append(
                        {"filename": safe_filename, "error": validation_msg}
                    )
                    continue

                duration = utils.get_audio_duration(staged_path)
                if duration is not None and duration <= 0:
                    upload_errors.append(
                        {
                            "filename": safe_filename,
                            "error": "File contains no audio (zero duration).",
                        }
                    )
                    continue

                # Trim before denoising: the model only ever sees the first
                # max_duration seconds, so there is no point enhancing the rest.
                if duration is not None and duration > max_duration:
                    trimmed_path = staged_path.with_name(
                        f"{staged_path.stem}_trimmed.wav"
                    )
                    ok, trim_msg, trimmed_duration = utils.trim_audio_file(
                        staged_path, trimmed_path, max_duration
                    )
                    if ok:
                        original_duration = duration
                        staged_path = trimmed_path
                        # Trimming always writes WAV, so an MP3 input changes
                        # extension; keep the stored name in step with the file.
                        safe_filename = f"{Path(safe_filename).stem}.wav"
                        duration = trimmed_duration
                        logger.info(
                            f"Trimmed '{safe_filename}' from {original_duration:.2f}s "
                            f"to {duration:.2f}s (max {max_duration}s)."
                        )
                        upload_warnings.append(
                            {
                                "filename": safe_filename,
                                "warning": (
                                    f"Clip was {original_duration:.1f}s, longer than the "
                                    f"{max_duration}s maximum — the first {duration:.1f}s "
                                    f"were kept."
                                ),
                            }
                        )
                    else:
                        upload_errors.append(
                            {"filename": safe_filename, "error": trim_msg}
                        )
                        continue

                # Denoise after trimming, so each clip is cleaned against its own
                # noise profile and only the audio being kept is processed.
                if denoise:
                    denoised_path = staged_path.with_name(
                        f"{staged_path.stem}_denoised.wav"
                    )
                    ok, denoise_msg = utils.denoise_audio_file(
                        staged_path, denoised_path
                    )
                    if ok:
                        staged_path = denoised_path
                        # Denoising always writes WAV, so an MP3 input changes
                        # extension; keep the stored name in step with the file.
                        safe_filename = f"{Path(safe_filename).stem}.wav"
                        denoised_any = True
                    elif not denoise_failure_reported:
                        # One notice per request, not one per file.
                        denoise_failure_reported = True
                        upload_warnings.append(
                            {"filename": safe_filename, "warning": denoise_msg}
                        )

                duration = utils.get_audio_duration(staged_path)
                staged_files.append((safe_filename, staged_path, duration))

            except Exception as e_upload:
                error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
                logger.error(error_msg, exc_info=True)
                upload_errors.append(
                    {"filename": file.filename, "error": str(e_upload)}
                )
            finally:
                await file.close()

        # A clip of unknown duration is treated as long enough: better to save it
        # untouched than to splice it into something the user did not ask for.
        too_short = [s for s in staged_files if s[2] is not None and s[2] < min_duration]
        keep_separate = [s for s in staged_files if s not in too_short]

        if len(too_short) >= 2:
            base_stem = Path(too_short[0][0]).stem
            destination_path = utils.get_unique_destination(
                ref_path, f"{base_stem}_combined.wav"
            )
            success, combine_msg, combined_duration = utils.concatenate_audio_files(
                [s[1] for s in too_short], destination_path, gap_ms
            )

            if success:
                combined_filename = destination_path.name
                combined_sources = [s[0] for s in too_short]
                uploaded_filenames_successfully.insert(0, combined_filename)
                logger.info(
                    f"Chained {len(too_short)} short clip(s) into '{combined_filename}': {combine_msg}"
                )

                # Enough short clips, each with a silence gap after it, can overrun
                # the maximum. Trim to fit rather than rejecting the batch.
                if combined_duration > max_duration:
                    ok, trim_msg, trimmed_duration = utils.trim_audio_file(
                        destination_path, destination_path, max_duration
                    )
                    if ok:
                        upload_warnings.append(
                            {
                                "filename": combined_filename,
                                "warning": (
                                    f"Chaining these {len(too_short)} clips produced "
                                    f"{combined_duration:.1f}s, over the {max_duration}s maximum — "
                                    f"the first {trimmed_duration:.1f}s were kept. Upload fewer "
                                    f"clips at a time to keep all of them."
                                ),
                            }
                        )
                        combined_duration = trimmed_duration
                    else:
                        upload_warnings.append(
                            {"filename": combined_filename, "warning": trim_msg}
                        )

                # Judge against speech content, not padded length: the inserted
                # silence lengthens the file without giving the model more voice
                # to work from, so it must not satisfy the minimum on its own.
                content_duration = sum(s[2] for s in too_short)
                if content_duration < min_duration:
                    upload_warnings.append(
                        {
                            "filename": combined_filename,
                            "warning": (
                                f"Combined reference is {combined_duration:.1f}s, but only "
                                f"{content_duration:.1f}s of that is speech — under the "
                                f"{min_duration:.0f}s recommended minimum. Cloning quality may suffer; "
                                f"add more clips and upload them together."
                            ),
                        }
                    )
            else:
                upload_errors.append(
                    {
                        "filename": ", ".join(s[0] for s in too_short),
                        "error": combine_msg,
                    }
                )
                keep_separate.extend(too_short)
        elif len(too_short) == 1:
            # Nothing to chain it with; save it, but say why it may disappoint.
            upload_warnings.append(
                {
                    "filename": too_short[0][0],
                    "warning": (
                        f"This clip is {too_short[0][2]:.1f}s, under the {min_duration:.0f}s recommended "
                        f"minimum for cloning. Upload it together with other short clips and they will be "
                        f"chained into one reference automatically."
                    ),
                }
            )
            keep_separate.extend(too_short)

        for safe_filename, staged_path, _duration in keep_separate:
            try:
                destination_path = utils.get_unique_destination(ref_path, safe_filename)
                shutil.move(str(staged_path), str(destination_path))
                logger.info(
                    f"Successfully saved uploaded reference file to: {destination_path}"
                )
                uploaded_filenames_successfully.append(destination_path.name)
            except Exception as e_move:
                error_msg = f"Error saving file '{safe_filename}': {str(e_move)}"
                logger.error(error_msg, exc_info=True)
                upload_errors.append({"filename": safe_filename, "error": str(e_move)})

    all_current_reference_files = utils.get_valid_reference_files()
    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_reference_files": all_current_reference_files,
        "combined_file": combined_filename,
        "combined_from": combined_sources,
        "denoised": denoised_any,
        "errors": upload_errors,
        "warnings": upload_warnings,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_reference completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    """
    Handles uploading of predefined voice files (.wav, .mp3).
    Validates files and saves them to the configured predefined voices path.
    """
    logger.info(f"Request to /upload_predefined_voice with {len(files)} file(s).")
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        safe_filename = utils.sanitize_filename(file.filename)
        destination_path = predefined_voices_path / safe_filename

        try:
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            # Basic validation (can be extended if predefined voices have specific requirements)
            is_valid, validation_msg = utils.validate_reference_audio(
                destination_path, max_duration_sec=None
            )  # No duration limit for predefined
            if not is_valid:
                logger.warning(
                    f"Uploaded predefined voice '{safe_filename}' failed basic validation: {validation_msg}. Deleting."
                )
                destination_path.unlink(missing_ok=True)
                upload_errors.append(
                    {"filename": safe_filename, "error": validation_msg}
                )
            else:
                uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    all_current_predefined_voices = (
        utils.get_predefined_voices()
    )  # Fetches formatted list
    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,  # List of raw filenames uploaded
        "all_predefined_voices": all_current_predefined_voices,  # Formatted list for UI
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)


# --- TTS Generation Endpoint ---


@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Required resource not found (e.g., voice file).",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks
):
    """
    Generates speech audio from text using specified parameters.
    Handles various voice modes (predefined, clone) and audio processing options.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(
        f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    audio_prompt_path_for_engine: Optional[Path] = None
    if request.voice_mode == "predefined":
        if not request.predefined_voice_id:
            raise HTTPException(
                status_code=400,
                detail="Missing 'predefined_voice_id' for 'predefined' voice mode.",
            )
        voices_dir = get_predefined_voices_path(ensure_absolute=True)
        try:
            potential_path = utils.safe_resolve_within(voices_dir, request.predefined_voice_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid predefined voice ID.")
        if not potential_path.is_file():
            logger.error(f"Predefined voice file not found: {potential_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Predefined voice file '{request.predefined_voice_id}' not found.",
            )
        audio_prompt_path_for_engine = potential_path
        logger.info(f"Using predefined voice: {request.predefined_voice_id}")

    elif request.voice_mode == "clone":
        if not request.reference_audio_filename:
            raise HTTPException(
                status_code=400,
                detail="Missing 'reference_audio_filename' for 'clone' voice mode.",
            )
        ref_dir = get_reference_audio_path(ensure_absolute=True)
        try:
            potential_path = utils.safe_resolve_within(ref_dir, request.reference_audio_filename)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid reference audio filename.")
        if not potential_path.is_file():
            logger.error(
                f"Reference audio file for cloning not found: {potential_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file '{request.reference_audio_filename}' not found.",
            )
        # Format only: an over-long reference is trimmed below, not rejected.
        is_valid, msg = utils.validate_reference_audio(potential_path)
        if not is_valid:
            raise HTTPException(
                status_code=400, detail=f"Invalid reference audio: {msg}"
            )

        # Applies the pitch/speed adjustment and the duration cap. The cap also
        # covers references that predate upload-time trimming, or that were copied
        # into the directory by hand. Results are cached beside the originals, so
        # this costs nothing after the first request for a given file and setting.
        max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
        prepared_path = utils.prepare_reference_audio(
            potential_path,
            max_dur,
            pitch_semitones=request.reference_pitch or 0.0,
            speed_factor=(
                request.reference_speed if request.reference_speed is not None else 1.0
            ),
        )
        if prepared_path is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Reference audio '{request.reference_audio_filename}' could not be "
                    f"prepared (duration cap {max_dur}s, pitch {request.reference_pitch or 0}, "
                    f"speed {request.reference_speed or 1})."
                ),
            )
        audio_prompt_path_for_engine = prepared_path
        logger.info(
            f"Using reference audio for cloning: {request.reference_audio_filename}"
        )

    perf_monitor.record("Parameters and voice path resolved")

    all_audio_segments_np: List[np.ndarray] = []
    final_output_sample_rate = (
        get_audio_sample_rate()
    )  # Target SR for the final output file
    engine_output_sample_rate: Optional[int] = (
        None  # SR from the TTS engine (e.g., 24000 Hz)
    )

    if request.split_text and len(request.text) > (
        request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5
    ):
        chunk_size_to_use = (
            request.chunk_size if request.chunk_size is not None else 120
        )
        logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size_to_use)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    # --- Streaming fork ---
    if request.stream:
        if request.output_format and request.output_format != "wav":
            logger.warning(
                f"stream=true: output_format '{request.output_format}' ignored; streaming always uses WAV."
            )

        speed_factor_stream = (
            request.speed_factor
            if request.speed_factor is not None
            else get_gen_default_speed_factor()
        )
        audio_prompt_str = (
            str(audio_prompt_path_for_engine) if audio_prompt_path_for_engine else None
        )
        temperature_val = (
            request.temperature if request.temperature is not None else get_gen_default_temperature()
        )
        exaggeration_val = (
            request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration()
        )
        cfg_weight_val = (
            request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight()
        )
        seed_val = request.seed if request.seed is not None else get_gen_default_seed()
        language_val = (
            request.language if request.language is not None else get_gen_default_language()
        )

        CROSSFADE_MS_STREAM = 20

        async def _stream_generator():
            loop = asyncio.get_running_loop()
            carry: Optional[np.ndarray] = None
            header_sent = False

            for i, chunk_text in enumerate(text_chunks):
                is_last = i == len(text_chunks) - 1
                logger.info(f"Streaming chunk {i+1}/{len(text_chunks)}...")

                audio_tensor, chunk_sr = await loop.run_in_executor(
                    None,
                    lambda c=chunk_text: engine.synthesize(
                        text=c,
                        audio_prompt_path=audio_prompt_str,
                        temperature=temperature_val,
                        exaggeration=exaggeration_val,
                        cfg_weight=cfg_weight_val,
                        seed=seed_val,
                        language=language_val,
                    ),
                )

                if audio_tensor is None or chunk_sr is None:
                    logger.error(f"Streaming TTS: engine returned None for chunk {i+1}; stopping stream.")
                    return

                # Streaming has no finished clip to stretch, so this stays
                # per-chunk. Chunks are sentence-sized, which is long enough
                # that restarting the stretcher at each boundary is inaudible.
                if speed_factor_stream != 1.0:
                    audio_tensor, _ = utils.apply_speed_factor(
                        audio_tensor, chunk_sr, speed_factor_stream
                    )

                audio_np = audio_tensor.cpu().numpy().squeeze().astype(np.float32)

                if not header_sent:
                    yield _create_wav_header(chunk_sr)
                    header_sent = True

                fade_samples = int(CROSSFADE_MS_STREAM / 1000 * chunk_sr)

                if carry is not None:
                    # Crossfade the held-back tail of the previous chunk with the head of this one
                    audio_np = _crossfade_with_overlap(carry, audio_np, fade_samples)
                    carry = None

                if not is_last and len(audio_np) > fade_samples:
                    carry = audio_np[-fade_samples:].copy()
                    yield _float32_to_pcm16(audio_np[:-fade_samples])
                else:
                    yield _float32_to_pcm16(audio_np)

            if carry is not None:
                yield _float32_to_pcm16(carry)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        stream_filename = utils.sanitize_filename(f"tts_stream_{timestamp_str}.wav")
        return StreamingResponse(
            _stream_generator(),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{stream_filename}"'},
        )
    # --- End streaming fork ---

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_tensor, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                audio_prompt_path=(
                    str(audio_prompt_path_for_engine)
                    if audio_prompt_path_for_engine
                    else None
                ),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else get_gen_default_temperature()
                ),
                exaggeration=(
                    request.exaggeration
                    if request.exaggeration is not None
                    else get_gen_default_exaggeration()
                ),
                cfg_weight=(
                    request.cfg_weight
                    if request.cfg_weight is not None
                    else get_gen_default_cfg_weight()
                ),
                seed=(
                    request.seed if request.seed is not None else get_gen_default_seed()
                ),
                language=(
                    request.language
                    if request.language is not None
                    else get_gen_default_language()
                ),
            )
            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_tensor is None or chunk_sr_from_engine is None:
                error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            current_processed_audio_tensor = chunk_audio_tensor

            # Speed is deliberately NOT applied here. Stretching every chunk
            # separately restarts the stretcher's analysis at each boundary and
            # leaves the stitched-in sentence pauses at their original length,
            # so a "faster" voice still pauses at normal speed. It is applied
            # once to the finished, stitched clip instead - see below.

            # ### MODIFICATION ###
            # All other processing is REMOVED from the loop.
            # We will process the final concatenated audio clip.
            processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
            all_audio_segments_np.append(processed_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=500, detail=error_detail)

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )
    try:
        # ### SMART AUDIO STITCHING ###
        # Local constants - adjust these values to tune stitching behavior
        SENTENCE_PAUSE_MS = 200  # Desired audible silence between sentences
        CROSSFADE_MS = 20  # Crossfade duration for smart mode (10-50ms recommended)
        SAFETY_FADE_MS = 3  # Minimal edge fade for fallback mode (2-5ms)
        ENABLE_DC_REMOVAL = False  # Set True if you hear low-frequency thumps
        DC_HIGHPASS_HZ = 15  # High-pass cutoff for DC removal
        PEAK_NORMALIZE_THRESHOLD = 0.99  # Normalize if peak exceeds this
        PEAK_NORMALIZE_TARGET = 0.95  # Target peak after normalization

        # Read smart stitching toggle from config (defaults to True)
        enable_smart_stitching = config_manager.get_bool(
            "audio_processing.enable_crossfade", True
        )

        # --- Sample rate validation ---
        if not engine_output_sample_rate or engine_output_sample_rate <= 0:
            logger.error(
                f"Invalid sample rate: {engine_output_sample_rate}, "
                "falling back to raw concatenation"
            )
            final_audio_np = (
                np.concatenate(all_audio_segments_np)
                if len(all_audio_segments_np) > 1
                else all_audio_segments_np[0]
            )

        elif len(all_audio_segments_np) == 1:
            # Single chunk - no stitching needed
            final_audio_np = all_audio_segments_np[0]
            logger.info("Single audio chunk - no stitching required")

        elif enable_smart_stitching:
            # --- Smart mode: true crossfading with silence insertion ---
            fade_samples = int(CROSSFADE_MS / 1000 * engine_output_sample_rate)

            # Calculate silence buffer with compensation for crossfade overlap
            # Each crossfade removes fade_samples from silence (one at each end)
            desired_silence_samples = int(
                SENTENCE_PAUSE_MS / 1000 * engine_output_sample_rate
            )
            silence_buffer_samples = desired_silence_samples + (fade_samples * 2)

            # Preprocess chunks: convert to float32 and optionally remove DC offset
            chunks = []
            for chunk in all_audio_segments_np:
                processed = chunk.astype(np.float32, copy=True)
                if ENABLE_DC_REMOVAL:
                    processed = _remove_dc_offset(
                        processed, engine_output_sample_rate, DC_HIGHPASS_HZ
                    )
                chunks.append(processed)

            # Start with first chunk
            result = chunks[0]

            # Stitch remaining chunks with crossfaded silence gaps
            for i in range(1, len(chunks)):
                # Create silence buffer (oversized to compensate for crossfade overlap)
                silence = np.zeros(silence_buffer_samples, dtype=np.float32)

                # Crossfade: current result → silence (speech fades into silence)
                result = _crossfade_with_overlap(result, silence, fade_samples)

                # Crossfade: result → next chunk (silence fades into speech)
                result = _crossfade_with_overlap(result, chunks[i], fade_samples)

            final_audio_np = result
            logger.info(
                f"Smart stitching applied: {len(chunks)} chunks, "
                f"{CROSSFADE_MS}ms crossfades, {SENTENCE_PAUSE_MS}ms pauses"
            )

        else:
            # --- Fallback mode: minimal safety edge fades, no silence ---
            fade_samples = int(SAFETY_FADE_MS / 1000 * engine_output_sample_rate)
            num_chunks = len(all_audio_segments_np)

            processed_chunks = []
            for i, chunk in enumerate(all_audio_segments_np):
                is_first = i == 0
                is_last = i == num_chunks - 1

                processed = _apply_edge_fades(
                    chunk,
                    fade_samples,
                    fade_in=(not is_first),  # No fade-in on first chunk
                    fade_out=(not is_last),  # No fade-out on last chunk
                )
                processed_chunks.append(processed)

            final_audio_np = np.concatenate(processed_chunks)
            logger.info(
                f"Safety edge fades applied: {num_chunks} chunks, "
                f"{SAFETY_FADE_MS}ms linear fades"
            )

        # --- Ensure float32 dtype for all code paths ---
        final_audio_np = final_audio_np.astype(np.float32, copy=False)

        # --- Normalize to prevent clipping ---
        peak_amplitude = np.abs(final_audio_np).max()
        if peak_amplitude > PEAK_NORMALIZE_THRESHOLD:
            final_audio_np = final_audio_np * (PEAK_NORMALIZE_TARGET / peak_amplitude)
            logger.warning(
                f"Audio normalized to prevent clipping (peak was {peak_amplitude:.3f})"
            )

        perf_monitor.record("Audio chunks stitched")

        # --- Global Audio Post-Processing (applied to complete stitched audio) ---
        if config_manager.get_bool("audio_processing.enable_silence_trimming", False):
            final_audio_np = utils.trim_lead_trail_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record("Global silence trim applied")

        if config_manager.get_bool(
            "audio_processing.enable_internal_silence_fix", False
        ):
            final_audio_np = utils.fix_internal_silence(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record("Global internal silence fix applied")

        if (
            config_manager.get_bool("audio_processing.enable_unvoiced_removal", False)
            and utils.PARSELMOUTH_AVAILABLE
        ):
            final_audio_np = utils.remove_long_unvoiced_segments(
                final_audio_np, engine_output_sample_rate
            )
            perf_monitor.record("Global unvoiced removal applied")

        # --- Warn about potentially conflicting settings ---
        if enable_smart_stitching and config_manager.get_bool(
            "audio_processing.enable_silence_trimming", False
        ):
            logger.warning(
                "Smart stitching adds sentence pauses, but silence trimming is enabled. "
                "Leading/trailing pauses may be removed."
            )
        # ### SMART AUDIO STITCHING END ###

    except ValueError as e_concat:
        logger.error(f"Audio concatenation/stitching failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio stitching error: {e_concat}"
        )

    # --- Talking speed (applied once, to the complete clip) ---
    # Placed after silence trimming and the internal-silence fix, whose
    # thresholds are tuned against natural-tempo speech, and after stitching so
    # that sentence pauses scale with the voice instead of staying fixed.
    speed_factor_to_use = (
        request.speed_factor
        if request.speed_factor is not None
        else get_gen_default_speed_factor()
    )
    if speed_factor_to_use != 1.0:
        final_audio_np = utils.apply_speed_factor_np(
            final_audio_np, engine_output_sample_rate, speed_factor_to_use
        )
        perf_monitor.record(f"Speed factor {speed_factor_to_use} applied to full clip")

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    # Include generation parameters in filename for easy comparison across presets
    temp_val = request.temperature if request.temperature is not None else get_gen_default_temperature()
    exag_val = request.exaggeration if request.exaggeration is not None else get_gen_default_exaggeration()
    cfg_val = request.cfg_weight if request.cfg_weight is not None else get_gen_default_cfg_weight()
    param_tag = f"T{temp_val:.1f}_E{exag_val:.1f}_W{cfg_val:.1f}".replace(".", "")
    suggested_filename_base = f"tts_output_{param_tag}_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    # Optional: Save to disk if enabled
    if config_manager.get_bool("audio_output.save_to_disk", False):
        output_dir = get_output_path(ensure_absolute=True)
        output_file_path = output_dir / download_filename
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "wb") as f:
                f.write(encoded_audio_bytes)
            if not output_file_path.exists() or output_file_path.stat().st_size < 100:
                logger.error(f"File save verification failed for {output_file_path}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save audio file to {output_file_path}",
                )
            logger.info(f"Audio saved to disk: {output_file_path}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to save audio to {output_file_path}: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to save audio file: {e}"
            )

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )

@app.get("/v1/audio/voices", tags=["llama-swap Compatible"])
# llama-swap, koboldcpp, and probably some more use this
async def openai_voices_endpoint(model: str = ""):
    logger.debug("Request for /v1/audio/voices.")
    try:
        return {"status": "ok", "voices": [voice["filename"] for voice in utils.get_predefined_voices()]}
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )

@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest):
    # Determine the audio prompt path based on the voice parameter
    predefined_voices_path = get_predefined_voices_path(ensure_absolute=True)
    reference_audio_path = get_reference_audio_path(ensure_absolute=True)
    try:
        voice_path_predefined = utils.safe_resolve_within(predefined_voices_path, request.voice)
        voice_path_reference = utils.safe_resolve_within(reference_audio_path, request.voice)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid voice parameter.")

    if voice_path_predefined.is_file():
        audio_prompt_path = voice_path_predefined
    elif voice_path_reference.is_file():
        audio_prompt_path = voice_path_reference
    else:
        raise HTTPException(
            status_code=404, detail=f"Voice file '{request.voice}' not found."
        )

    # The model's reference limit applies whichever directory the voice came from.
    max_dur = config_manager.get_int("audio_output.max_reference_duration_sec", 30)
    trimmed_prompt_path = utils.prepare_reference_audio(audio_prompt_path, max_dur)
    if trimmed_prompt_path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Voice file '{request.voice}' is longer than the {max_dur}s maximum "
                f"and could not be trimmed."
            ),
        )
    audio_prompt_path = trimmed_prompt_path

    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    try:
        seed_to_use = (
            request.seed if request.seed is not None else get_gen_default_seed()
        )

        # Split long text into chunks for better quality (same as /tts endpoint)
        DEFAULT_CHUNK_SIZE = 120
        text_chunks = utils.chunk_text_by_sentences(request.input_, DEFAULT_CHUNK_SIZE)
        if not text_chunks:
            raise HTTPException(
                status_code=400, detail="Text processing resulted in no usable chunks."
            )

        logger.info(
            f"OpenAI speech: processing {len(text_chunks)} chunk(s) for input of {len(request.input_)} chars"
        )

        all_audio_segments_np: List[np.ndarray] = []
        engine_sr: Optional[int] = None

        for i, chunk_text in enumerate(text_chunks):
            chunk_seed = seed_to_use + i if seed_to_use is not None and seed_to_use >= 0 else seed_to_use

            audio_tensor, sr = engine.synthesize(
                text=chunk_text,
                audio_prompt_path=str(audio_prompt_path),
                temperature=get_gen_default_temperature(),
                exaggeration=get_gen_default_exaggeration(),
                cfg_weight=get_gen_default_cfg_weight(),
                seed=chunk_seed,
                language=request.language or get_gen_default_language(),
            )

            if audio_tensor is None or sr is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"TTS engine failed to synthesize audio for chunk {i+1}.",
                )

            if engine_sr is None:
                engine_sr = sr

            # Speed is applied once to the stitched clip below, not per chunk.
            chunk_np = audio_tensor.cpu().numpy().squeeze().astype(np.float32)
            all_audio_segments_np.append(chunk_np)

        # Stitch chunks together with crossfading
        if len(all_audio_segments_np) == 1:
            final_audio_np = all_audio_segments_np[0]
        else:
            CROSSFADE_MS = 20
            SENTENCE_PAUSE_MS = 200
            fade_samples = int(CROSSFADE_MS / 1000 * engine_sr)
            silence_buffer_samples = int(SENTENCE_PAUSE_MS / 1000 * engine_sr) + (fade_samples * 2)

            result = all_audio_segments_np[0].astype(np.float32)
            for seg in all_audio_segments_np[1:]:
                seg = seg.astype(np.float32)
                silence = np.zeros(silence_buffer_samples, dtype=np.float32)
                result = _crossfade_with_overlap(result, silence, fade_samples)
                result = _crossfade_with_overlap(result, seg, fade_samples)
            final_audio_np = result
            logger.info(
                f"OpenAI speech: stitched {len(all_audio_segments_np)} chunks with {CROSSFADE_MS}ms crossfades"
            )

        # Talking speed, applied once to the stitched clip so that the sentence
        # pauses scale with the voice and the stretcher runs a single pass.
        if request.speed != 1.0:
            final_audio_np = utils.apply_speed_factor_np(
                final_audio_np, engine_sr, request.speed
            )

        # Normalize to prevent clipping
        peak = np.abs(final_audio_np).max()
        if peak > 0.99:
            final_audio_np = final_audio_np * (0.95 / peak)

        encoded_audio = utils.encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )

        if encoded_audio is None:
            raise HTTPException(status_code=500, detail="Failed to encode audio.")

        media_type = f"audio/{request.response_format}"

        # Optional: Save to disk if enabled
        if config_manager.get_bool("audio_output.save_to_disk", False):
            output_dir = get_output_path(ensure_absolute=True)
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            download_filename = f"openai_tts_{timestamp_str}.{request.response_format}"
            output_file_path = output_dir / download_filename
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_file_path, "wb") as f:
                    f.write(encoded_audio)
                if (
                    not output_file_path.exists()
                    or output_file_path.stat().st_size < 100
                ):
                    logger.error(
                        f"File save verification failed for {output_file_path}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to save audio file to {output_file_path}",
                    )
                logger.info(
                    f"OpenAI-compatible audio saved to disk: {output_file_path}"
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to save audio to {output_file_path}: {e}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to save audio file: {e}"
                )

        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()
    ssl_kwargs = get_ssl_config()
    protocol = "https" if ssl_kwargs else "http"

    logger.info(f"Starting TTS Server directly on {protocol}://{server_host}:{server_port}")
    logger.info(
        f"API documentation will be available at {protocol}://{server_host}:{server_port}/docs"
    )
    logger.info(f"Web UI will be available at {protocol}://{server_host}:{server_port}/")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
        **ssl_kwargs,
    )
