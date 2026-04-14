"""
TTS model initialization and management
"""

import os
import asyncio
import torch
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from safetensors.torch import load_file as load_safetensors
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device


def _load_multilingual_with_resize(ckpt_dir: Path, device: str) -> ChatterboxMultilingualTTS:
    """
    Load multilingual model from local directory, handling vocab size mismatches
    from finetuned checkpoints (e.g., extended tokenizer with extra tokens).
    """
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.mtl_tts import Conditionals

    # Load voice encoder
    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
    ve = ve.to(device).eval()

    # Load T3 state dict to check vocab size
    t3_state = load_safetensors(ckpt_dir / "t3_23lang.safetensors")
    ckpt_vocab_size = t3_state["text_emb.weight"].shape[0]

    # Determine tokenizer — use tokenizer.json (extended) if present, else mtl_tokenizer.json
    std_tok_path = ckpt_dir / "tokenizer.json"
    mtl_tok_path = ckpt_dir / "mtl_tokenizer.json"
    if std_tok_path.exists():
        tokenizer = MTLTokenizer(str(std_tok_path))
    elif mtl_tok_path.exists():
        tokenizer = MTLTokenizer(str(mtl_tok_path))
    else:
        raise FileNotFoundError(f"No tokenizer found in {ckpt_dir}")

    # Create T3 with a config matching the checkpoint's vocab size
    hp = T3Config(text_tokens_dict_size=ckpt_vocab_size)
    t3 = T3(hp=hp)
    t3.load_state_dict(t3_state, strict=True)
    t3 = t3.to(device).eval()

    print(f"  T3 loaded with vocab size {ckpt_vocab_size}")

    # Load S3Gen
    s3gen = S3Gen()
    s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True), strict=False)
    s3gen = s3gen.to(device).eval()

    # Load conditionals if present
    conds = None
    if (ckpt_dir / "conds.pt").exists():
        conds = Conditionals.load(ckpt_dir / "conds.pt").to(device)

    return ChatterboxMultilingualTTS(t3, s3gen, ve, tokenizer, device, conds=conds)

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages
    
    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None
            
            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)
            
            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')
            
            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file
        
        # Determine if we should use multilingual model
        use_multilingual = Config.USE_MULTILINGUAL_MODEL
        
        _initialization_progress = "Loading TTS model (this may take a while)..."
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        local_dir = Config.MODEL_LOCAL_DIR

        if use_multilingual:
            print(f"Loading Chatterbox Multilingual TTS model...")
            if local_dir:
                print(f"Using local weights from: {local_dir}")
                _model = await loop.run_in_executor(
                    None,
                    lambda: _load_multilingual_with_resize(Path(local_dir), _device)
                )
            else:
                _model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxMultilingualTTS.from_pretrained(device=_device)
                )
            _is_multilingual = True
            _supported_languages = SUPPORTED_LANGUAGES.copy()
            print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
        else:
            print(f"Loading standard Chatterbox TTS model...")
            if local_dir:
                print(f"Using local weights from: {local_dir}")
                _model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxTTS.from_local(Path(local_dir), device=_device)
                )
            else:
                _model = await loop.run_in_executor(
                    None,
                    lambda: ChatterboxTTS.from_pretrained(device=_device)
                )
            _is_multilingual = False
            _supported_languages = {"en": "English"}  # Standard model only supports English
            print(f"✓ Standard model initialized (English only)")
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    return {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state
    }