"""
HuggingFace Audio Datasets Integration for Immersive Audio Generation

This module provides functionality to use HuggingFace audio datasets as sound sources
for the immersive audio generation system. It integrates with the existing CLAP-based
search system to enable semantic retrieval from thousands of audio samples.

Popular HuggingFace Audio Datasets:
- ESC-50: Environmental Sound Classification (2,000 labeled sounds)
- FSD50K: Freesound Dataset (51,197 sounds)
- AudioSet: Large-scale audio dataset (2.1M clips)
- LibriSpeech: Speech corpus (1,000 hours)
- UrbanSound8K: Urban environmental sounds (8,732 clips)
- GTZAN: Music genre classification (1,000 songs)
- BBC Sound Effects: BBC's sound effect library (20,000+ sounds)
"""

import os
import glob
import ctypes
from helpers import preload_nvidia_libs

# Preload NVIDIA libraries before other imports
preload_nvidia_libs()

import torch
import faiss
import pickle
import os
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from transformers import ClapAudioModelWithProjection, ClapProcessor, ClapTextModelWithProjection, AutoTokenizer
from typing import List, Dict, Optional
from config import config


def init_search(index_file: str = None, metadata_file: str = None):
    """
    Initialize search with CLAP model, tokenizer, and FAISS index.
    
    Parameters:
    -----------
    index_file : str
        Path to the FAISS index file (defaults to config value)
    metadata_file : str
        Path to the metadata pickle file (defaults to config value)
    
    Returns:
    --------
    model, tokenizer, device, index, metadata
    """
    if index_file is None:
        index_file = config.database.hf_index_file
    if metadata_file is None:
        metadata_file = config.database.hf_metadata_file
    
    model = ClapTextModelWithProjection.from_pretrained(config.database.clap_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.database.clap_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    index = faiss.read_index(index_file)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    return model, tokenizer, device, index, metadata


# Popular audio datasets on HuggingFace with their configurations
POPULAR_DATASETS = {
    'esc50': {
        'name': 'ashraq/esc50',
        'config': None,
        'description': '2,000 environmental sounds in 50 categories',
        'audio_column': 'audio',
        'label_column': 'category',
        'sample_rate': 44100
    },
    'fsd50k': {
        'name': 'Chand0320/fsd50k_hf',
        'config': None,
        'description': '51,197 sounds from Freesound',
        'audio_column': 'audio',
        'label_column': 'label',
        'sample_rate': 44100
    },
    'urbansound8k': {
        'name': 'danavery/urbansound8K',
        'config': None,
        'description': '8,732 urban environmental sounds',
        'audio_column': 'audio',
        'label_column': 'class',
        'sample_rate': 44100
    },
    'bbc_sounds': {
        'name': 'CLAPv2/BBCSoundEffects',
        'config': None,
        'description': 'BBC sound effects library',
        'audio_column': 'audio',
        'label_column': 'label',
        'sample_rate': 48000
    },
    'gtzan': {
        'name': 'storylinez/gtzan-music-genre-dataset',
        'config': None,
        'description': '1,000 music clips in 10 genres',
        'audio_column': 'audio',
        'label_column': 'genre',
        'sample_rate': 22050
    },
    'speech_commands': {
        'name': 'google/speech_commands',
        'config': None,
        'description': '105,829 spoken words',
        'audio_column': 'audio',
        'label_column': 'label',
        'sample_rate': 16000
    }
}


def create_hf_embeddings(
    dataset_names: List[str],
    max_samples: Optional[int] = None,
    cache_dir: str = None
) -> Dict:
    """
    Create audio embeddings from HuggingFace datasets and build FAISS index.
    
    Parameters:
    -----------
    dataset_names : List[str]
        List of dataset keys from POPULAR_DATASETS to load (e.g., ['esc50', 'urbansound8k'])
        Or custom HuggingFace dataset paths
    max_samples : Optional[int]
        Maximum number of samples to process from each dataset (None for all)
    cache_dir : str
        Directory to cache downloaded datasets (defaults to config.database.hf_cache_dir)
    
    Returns:
    --------
    Dict with statistics about created embeddings
    """
    
    if cache_dir is None:
        cache_dir = config.database.hf_cache_dir
    
    # Create directory for audio files
    audio_dir = 'hf_audio'
    os.makedirs(audio_dir, exist_ok=True)
    
    print(f"Loading CLAP model...")
    model = ClapAudioModelWithProjection.from_pretrained(config.database.clap_model_id)
    processor = ClapProcessor.from_pretrained(config.database.clap_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    metadata = []
    embeddings_list = []
    index = faiss.IndexFlatL2(512)
    
    stats = {
        'datasets_processed': [],
        'total_samples': 0,
        'successful_embeddings': 0,
        'failed_embeddings': 0
    }
    
    for dataset_key in dataset_names:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_key}")
        print(f"{'='*60}")
        
        # Check if it's a predefined dataset or custom path
        if dataset_key in POPULAR_DATASETS:
            dataset_info = POPULAR_DATASETS[dataset_key]
            dataset_name = dataset_info['name']
            dataset_config = dataset_info['config']
            audio_column = dataset_info['audio_column']
            label_column = dataset_info['label_column']
        else:
            # Custom HuggingFace dataset path
            dataset_name = dataset_key
            dataset_config = None
            audio_column = 'audio'  # Default assumption
            label_column = 'label'  # Default assumption
        
        try:
            # Load dataset with streaming for large datasets
            dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                split='train',
                streaming=False,
                cache_dir=cache_dir
            )
            
            # Resample to 48kHz if needed
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=48000))
            
            dataset_samples = len(dataset)
            if max_samples:
                dataset_samples = min(dataset_samples, max_samples)
                dataset = dataset.select(range(dataset_samples))
            
            dataset_stats = {
                'name': dataset_name,
                'total_samples': dataset_samples,
                'processed': 0,
                'successful': 0,
                'failed': 0
            }
            
            print(f"Dataset loaded: {dataset_samples} samples")
            
            for idx, item in enumerate(dataset):
                print(f"Processing sample {idx + 1}/{dataset_samples}...", end='\r')
                
                try:
                    # Get audio data
                    audio_data = item[audio_column]
                    s = audio_data['array']
                    fs = audio_data['sampling_rate']
                    
                    # Ensure mono
                    if len(s.shape) > 1:
                        s = s.mean(axis=1)
                    
                    # Get label/description
                    label = item.get(label_column, 'unknown')
                    if isinstance(label, int):
                        label = str(label)
                    
                    # Generate embedding
                    inputs = processor(audio=s, return_tensors="pt", sampling_rate=fs)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    embedding = model(**inputs).audio_embeds.detach().cpu().numpy().squeeze()
                    
                    if embedding is not None:
                        embeddings_list.append(embedding)
                        
                        # Write audio to disk instead of storing in pickle
                        audio_filename = f'{dataset_key}_{idx}.wav'
                        audio_path = os.path.join(audio_dir, audio_filename)
                        sf.write(audio_path, s, fs)
                        
                        # Store metadata with file path reference
                        metadata.append({
                            'file': f"hf://{dataset_name}/{idx}",
                            'dataset': dataset_name,
                            'sample_idx': idx,
                            'label': label,
                            'source': 'huggingface',
                            'audio_file': audio_path,
                            'sampling_rate': fs
                        })
                        
                        dataset_stats['successful'] += 1
                        stats['successful_embeddings'] += 1
                    else:
                        dataset_stats['failed'] += 1
                        stats['failed_embeddings'] += 1
                    
                    dataset_stats['processed'] += 1
                    
                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    dataset_stats['failed'] += 1
                    stats['failed_embeddings'] += 1
                    continue
            
            print(f"\nDataset {dataset_name} completed:")
            print(f"  Processed: {dataset_stats['processed']}")
            print(f"  Successful: {dataset_stats['successful']}")
            print(f"  Failed: {dataset_stats['failed']}")
            
            stats['datasets_processed'].append(dataset_stats)
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue
    
    # Build FAISS index
    if embeddings_list:
        print(f"\nBuilding FAISS index with {len(embeddings_list)} embeddings...")
        index.add(np.vstack(embeddings_list).astype('float32'))
        
        # Save index
        print(f"Saving FAISS index to {config.database.hf_index_file}...")
        faiss.write_index(index, config.database.hf_index_file)
        
        # Save metadata
        print(f"Saving metadata to {config.database.hf_metadata_file}...")
        with open(config.database.hf_metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        stats['total_samples'] = len(metadata)
        
        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'='*60}")
        print(f"Total datasets processed: {len(stats['datasets_processed'])}")
        print(f"Total embeddings created: {stats['successful_embeddings']}")
        print(f"Failed embeddings: {stats['failed_embeddings']}")
        print(f"Index saved to: {config.database.hf_index_file}")
        print(f"Metadata saved to: {config.database.hf_metadata_file}")
        print(f"Cache directory: {cache_dir}")
    else:
        print("No embeddings were created!")
    
    return stats



def search_hf_audio(
    text: str,
    model,
    tokenizer,
    device,
    index,
    metadata,
    k: int = None
) -> List[Dict]:
    """
    Search for similar audio in HuggingFace datasets.
    
    Parameters:
    -----------
    text : str
        Text query describing the desired sound
    model : ClapTextModelWithProjection
        CLAP text model
    tokenizer : AutoTokenizer
        CLAP tokenizer
    device : torch.device
        Device to run model on
    index : faiss.Index
        FAISS index
    metadata : list
        List of metadata entries
    k : int
        Number of results to return (defaults to config.database.search_results)
    
    Returns:
    --------
    List of results with metadata
    """
    
    if k is None:
        k = config.database.search_results
    
    # Generate text embedding
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    text_embedding = model(**inputs).text_embeds.detach().cpu().numpy().squeeze()
    text_embedding = text_embedding.reshape(1, -1).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(text_embedding, k)
    
    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < len(metadata):
            item = metadata[idx]
            results.append({
                'rank': i + 1,
                'file': item['file'],
                'dataset': item['dataset'],
                'label': item['label'],
                'distance': float(dist),
                'metadata': item
            })
    
    return results


def get_audio_from_metadata(metadata_item: Dict, duration: float, auto_trim: bool = False, trim_threshold_db: float = -30) -> tuple:
    """
    Get audio data from metadata item.
    Loads audio from disk (stored during create_hf_embeddings) rather than from pickle.
    
    Parameters:
    -----------
    metadata_item : Dict
        Metadata entry from search results
    duration : float
        Desired duration in seconds
    auto_trim : bool
        If True, trim quiet segments before looping (default: False)
    trim_threshold_db : float
        Energy threshold for auto_trim in dB (default: -30)
    
    Returns:
    --------
    audio_array, sampling_rate
    """
    
    audio_file = metadata_item['audio_file']
    fs = metadata_item['sampling_rate']
    
    # Load audio from disk
    s, _ = sf.read(audio_file, dtype='float32')
    if s.ndim > 1:
        s = s.mean(axis=1)
    
    # Auto-trim quiet segments BEFORE looping (if requested)
    if auto_trim:
        s = _trim_quiet_segments(s, fs, trim_threshold_db)
    
    # Tile or trim to desired duration with crossfade at loop points
    samples_needed = int(duration * fs)
    crossfade_samples = int(0.005 * fs)  # 5ms crossfade
    
    if len(s) < samples_needed:
        # Need to loop - apply crossfade at loop boundary
        n_repeats = int(np.ceil(samples_needed / len(s)))
        
        # Apply fade out at end of original
        if len(s) > crossfade_samples:
            s_faded = s.copy()
            fade_out = np.linspace(1, 0, crossfade_samples)
            s_faded[-crossfade_samples:] *= fade_out
            
            # Apply fade in at start
            fade_in = np.linspace(0, 1, crossfade_samples)
            s_faded[:crossfade_samples] *= fade_in
            
            # Tile the faded version
            s = np.tile(s_faded, n_repeats)[:samples_needed]
        else:
            s = np.tile(s, n_repeats)[:samples_needed]
    else:
        s = s[:samples_needed]
    
    return s, fs


def _trim_quiet_segments(sig, fs, threshold_db):
    """Trim quiet segments from a signal, preserving total duration via looping.
    
    This is an internal function used by get_audio_from_metadata.
    Trimming happens before looping, so the final duration is always exact.
    Applies short crossfades at splice points to prevent clicks.
    """
    # Calculate energy in 100ms windows
    window_size = int(0.1 * fs)
    n_windows = len(sig) // window_size
    
    if n_windows == 0:
        return sig
    
    active_windows = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = sig[start:end]
        rms = np.sqrt(np.mean(window ** 2))
        energy_db = 20 * np.log10(rms + 1e-15)
        if energy_db > threshold_db:
            active_windows.append((start, end))
    
    if not active_windows:
        return sig  # No active segments, return original
    
    # Merge consecutive windows
    merged = []
    for start, end in active_windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    
    # Concatenate active segments with short crossfades to prevent clicks
    crossfade_samples = int(0.005 * fs)  # 5ms crossfade
    segments = []
    for i, (start, end) in enumerate(merged):
        seg = sig[start:end].copy()
        
        # Apply fade in at the start of each segment (except first)
        if i > 0 and len(seg) > crossfade_samples:
            fade_in = np.linspace(0, 1, crossfade_samples)
            seg[:crossfade_samples] *= fade_in
        
        # Apply fade out at the end of each segment (except last)
        if i < len(merged) - 1 and len(seg) > crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            seg[-crossfade_samples:] *= fade_out
        
        segments.append(seg)
    
    return np.concatenate(segments) if segments else sig


# Default text corpus for sound classification via CLAP embeddings
DEFAULT_TEXT_CORPUS = [
    # Animals
    "dog barking", "dog", "barking dog", "small dog barking", "large dog barking",
    "cat meowing", "cat", "meowing cat", "cat purring",
    "bird chirping", "bird", "chirping bird", "songbird", "sparrow",
    "horse neighing", "horse", "neighing",
    "cow mooing", "cow", "mooing",
    "sheep bleating", "sheep", "goat",
    "pig grunting", "pig", "oinking",
    "chicken clucking", "chicken", "rooster crowing",
    "insect buzzing", "bee buzzing", "cricket chirping", "fly",
    "wolf howling", "wolf", "howling",
    "lion roaring", "lion", "tiger", "wild animal",
    
    # Nature - Weather
    "rain falling", "rain", "light rain", "heavy rain", "rainstorm",
    "thunder", "thunderstorm", "lightning strike",
    "wind blowing", "wind", "strong wind", "gentle breeze", "howling wind",
    "waves crashing", "ocean waves", "sea", "beach waves",
    "water flowing", "river", "stream", "babbling brook", "waterfall",
    "fire crackling", "fire", "campfire", "bonfire",
    "snow", "hail", "ice",
    
    # Nature - Environment
    "forest ambience", "forest", "woods", "jungle",
    "leaves rustling", "leaves", "branches",
    "grass", "meadow", "field",
    
    # Urban - Vehicles
    "car engine", "car", "engine", "motor", "car passing",
    "motorcycle", "motorbike", "scooter",
    "truck", "truck engine", "heavy vehicle",
    "bus", "bus engine",
    "train", "train passing", "subway", "metro",
    "airplane", "airplane flying", "jet", "aircraft", "helicopter",
    "boat", "ship", "motorboat", "speedboat",
    "siren", "police siren", "ambulance", "fire truck",
    "car horn", "horn", "honking",
    
    # Urban - Traffic
    "traffic", "city traffic", "busy street", "highway",
    "construction", "jackhammer", "drill", "machinery",
    
    # Urban - General
    "crowd", "crowd talking", "people talking", "busy crowd",
    "footsteps", "walking", "running", "footsteps on pavement",
    "door opening", "door closing", "door", "door slam",
    "doorbell", "doorbell ringing",
    "elevator", "elevator ding",
    "telephone ringing", "phone", "telephone",
    "alarm", "alarm clock", "smoke alarm", "car alarm",
    
    # Human - Voice
    "speech", "talking", "conversation", "whispering",
    "laughter", "laughing", "giggle",
    "scream", "screaming", "shout", "yelling",
    "crying", "sobbing", "baby crying",
    "singing", "humming", "whistling",
    "coughing", "sneezing", "breathing",
    "applause", "clapping", "cheering",
    
    # Human - Activities
    "eating", "chewing", "drinking", "swallowing",
    "typing", "keyboard", "computer keyboard",
    "writing", "pen", "pencil",
    "clapping", "snapping fingers",
    
    # Music - Instruments
    "piano", "piano playing", "keyboard",
    "guitar", "guitar strumming", "acoustic guitar", "electric guitar",
    "drums", "drum beat", "percussion", "drumming",
    "violin", "string instrument", "cello",
    "bass", "bass guitar", "bass line",
    "synthesizer", "synth", "electronic",
    "trumpet", "brass", "saxophone", "flute",
    
    # Music - Genres
    "music", "classical music", "rock music", "jazz", "electronic music",
    "ambient music", "background music",
    
    # Sound Effects
    "explosion", "gunshot", "gunfire", "cannon",
    "glass breaking", "glass shattering", "smash",
    "metal clang", "metal", "clanking", "banging",
    "wood creaking", "creaking", "squeaking",
    "bell", "church bell", "ringing bell",
    "clock ticking", "clock", "ticking",
    "beep", "electronic beep", "tone",
    "static", "white noise", "static noise",
    "sci-fi", "futuristic", "spaceship", "laser",
    
    # Horror/Thriller
    "horror ambience", "creepy", "scary", "eerie",
    "heartbeat", "heart beating", "pulse",
    "chainsaw", "power tool",
    "monster", "creature", "alien",
    
    # Impact/Hits
    "impact", "hit", "punch", "kick", "slap",
    "thud", "dull impact",
    "crash", "collision", "smash",
    
    # Ambience - Indoor
    "room tone", "silence", "quiet room",
    "office ambience", "office", "workplace",
    "restaurant", "cafe", "bar",
    "kitchen", "cooking", "dishes",
    "bathroom", "shower", "water running",
    
    # Ambience - Outdoor
    "city ambience", "urban", "downtown",
    "park", "playground", "children playing",
    "stadium", "sports crowd", "audience",
    "farm", "barn", "countryside",
    
    # Electronic/Digital
    "computer", "fan", "air conditioning", "refrigerator",
    "printer", "scanner", "photocopier",
    
    # Sports
    "basketball", "bouncing ball", "tennis",
    "football", "sports crowd",
    "swimming", "diving", "splash",
    
    # Other
    "fart", "burp", "body sounds",
    "smoking", "lighter", "match",
    "candle", "blowing out",
]


def create_text_corpus_index(
    text_corpus: list = None,
    output_file: str = None,
    model=None,
    tokenizer=None,
    device=None
):
    """
    Create FAISS index from text descriptions using CLAP text encoder.
    
    Parameters:
    -----------
    text_corpus : list
        List of text descriptions. Defaults to DEFAULT_TEXT_CORPUS.
    output_file : str
        Output path for FAISS index. Defaults to config value.
    model : ClapTextModelWithProjection
        CLAP text model (will load if not provided)
    tokenizer : AutoTokenizer
        CLAP tokenizer (will load if not provided)
    device : str
        Device to run model on
    
    Returns:
    --------
    FAISS index, text_corpus
    """
    if text_corpus is None:
        text_corpus = DEFAULT_TEXT_CORPUS
    
    if output_file is None:
        output_file = config.database.text_corpus_index
    
    # Load model if not provided
    if model is None:
        from transformers import ClapTextModelWithProjection, AutoTokenizer
        model = ClapTextModelWithProjection.from_pretrained(config.database.clap_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.database.clap_model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    
    print(f"Computing embeddings for {len(text_corpus)} text descriptions...")
    
    # Compute embeddings in batches
    batch_size = 64
    embeddings = []
    
    for i in range(0, len(text_corpus), batch_size):
        batch = text_corpus[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            batch_embeddings = model(**inputs).text_embeds.cpu().numpy()
        embeddings.append(batch_embeddings)
        print(f"  Processed {min(i+batch_size, len(text_corpus))}/{len(text_corpus)}", end='\r')
    
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, output_file)
    
    # Save corpus
    corpus_file = output_file.replace('.faiss', '.pkl')
    with open(corpus_file, 'wb') as f:
        pickle.dump(text_corpus, f)
    
    print(f"\nText corpus index saved to {output_file}")
    print(f"Text corpus saved to {corpus_file}")
    
    return index, text_corpus


def load_text_corpus_index(index_file: str = None, corpus_file: str = None):
    """
    Load pre-computed text corpus FAISS index.
    
    Parameters:
    -----------
    index_file : str
        Path to FAISS index file
    corpus_file : str
        Path to corpus pickle file
    
    Returns:
    --------
    index, text_corpus or (None, None) if not found
    """
    if index_file is None:
        index_file = config.database.text_corpus_index
    if corpus_file is None:
        corpus_file = index_file.replace('.faiss', '.pkl')
    
    try:
        index = faiss.read_index(index_file)
        with open(corpus_file, 'rb') as f:
            text_corpus = pickle.load(f)
        return index, text_corpus
    except (FileNotFoundError, Exception):
        return None, None


def list_available_datasets():
    """
    List available HuggingFace audio datasets with descriptions.
    
    Returns:
    --------
    Dict of available datasets
    """
    
    print("\nAvailable HuggingFace Audio Datasets:")
    print("=" * 70)
    
    for key, info in POPULAR_DATASETS.items():
        print(f"\n{key}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Sample rate: {info['sample_rate']} Hz")
    
    return POPULAR_DATASETS


if __name__ == "__main__":
    # Example usage

    # This creates a FAISS index of text descriptions for labeling sounds during mix analysis.
    create_text_corpus_index()
    
    # List available datasets
    list_available_datasets()
    
    # Create embeddings from HuggingFace datasets
    print("\n" + "="*70)
    print("Creating embeddings from HuggingFace datasets...")
    print("="*70 + "\n")
    
    stats = create_hf_embeddings(
        dataset_names=['fsd50k'],
        max_samples=2000
    )
    
    # Test search
    if stats['successful_embeddings'] > 0:
        print("\n" + "="*70)
        print("Testing audio search...")
        print("="*70 + "\n")
        
        model, tokenizer, device, index, metadata = init_search(index_file=config.database.hf_index_file, metadata_file=config.database.hf_metadata_file)
        results = search_hf_audio('barking dog', model, tokenizer, device, index, metadata, k=3)
        
        for result in results:
            print(f"\nRank {result['rank']}: {result['label']}")
            print(f"  Dataset: {result['dataset']}")
            print(f"  Distance: {result['distance']:.4f}")
            print(f"  File: {result['file']}")
