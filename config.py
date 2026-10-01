"""
Configuration settings for the immersive audio generation system.
"""

from dataclasses import dataclass, field
from typing import Dict
import os
from pathlib import Path


@dataclass
class OllamaConfig:
    """Ollama/LLM settings."""
    model: str = 'hf.co/Qwen/Qwen3-32B-GGUF:Q5_K_M'
    temperature: float = 0.5  # 0.3-0.4 for more reliable tool call formatting
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.8  # 0.5-0.8 for robustness (high value may cause the LLM to abandon its plan)
    max_iterations: int = 100


@dataclass
class AudioConfig:
    """Audio processing settings."""
    sample_rate: int = 48000
    default_duration: float = 10.0
    block_size: int = 2048
    headroom_db: float = 0.9


@dataclass
class DatabaseConfig:
    """Database settings for audio search."""
    clap_model_id: str = 'laion/clap-htsat-fused'
    local_index_file: str = "audio_faiss.index"
    local_metadata_file: str = "audio_metadata.pkl"
    hf_index_file: str = "hf_audio_faiss.index"
    hf_metadata_file: str = "hf_audio_metadata.pkl"
    hf_cache_dir: str = "./hf_datasets_cache"
    search_results: int = 10  # k value for FAISS search, if >1 result is randomized based on inverse distance
    text_corpus_index: str = "text_corpus.faiss"  # FAISS index for text descriptions
    similarity_threshold: float = 0.3  # Cosine distance threshold for sound grouping. Range: 0 (identical) to 2 (opposite). Typical: 0.1 (strict) to 0.5 (relaxed).
    max_analysis_sounds: int = 10  # Maximum number of sounds to output from analysis (primary control for output size)


@dataclass
class LayoutConfig:
    """Speaker layout mappings."""
    layout_map: Dict[str, str] = field(default_factory=lambda: {
        'stereo': '2.0.0',
        'surround': '5.1.0',
        'immersive': '7.1.4',
    })


@dataclass
class Config:
    """Main configuration container."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config with overrides from environment variables."""
        config = cls()
        
        # Ollama overrides
        model = os.getenv('OLLAMA_MODEL')
        if model:
            config.ollama.model = model
        temperature = os.getenv('OLLAMA_TEMPERATURE')
        if temperature:
            config.ollama.temperature = float(temperature)
        max_iter = os.getenv('OLLAMA_MAX_ITERATIONS')
        if max_iter:
            config.ollama.max_iterations = int(max_iter)
        
        # Audio overrides
        sample_rate = os.getenv('AUDIO_SAMPLE_RATE')
        if sample_rate:
            config.audio.sample_rate = int(sample_rate)
        
        # Database overrides
        clap_model = os.getenv('CLAP_MODEL_ID')
        if clap_model:
            config.database.clap_model_id = clap_model
        
        return config


# Global config instance
config = Config.from_env()
