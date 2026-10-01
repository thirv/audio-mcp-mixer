import os
import random
import ast
import numpy as np
import soundfile as sf
from helpers import (
    check_overwrite, check_inputfile,
    strip_wav, get_text_labels,
    group_sounds_with_movement, group_sounds_by_embedding,
    apply_energy_envelope, analyze_temporal_continuity,
    cosine_distance
)
# NOTE: `torch` and the `database_hf` search helpers are imported lazily inside
# _get_clap() (see below) so that `import tools` stays fast. They are bound as
# module globals there; every tool that uses them calls _get_clap() first.
from typing import List, Optional



import logging

# Module-level logger: goes to stderr so messages never corrupt MCP stdio JSON-RPC
logger = logging.getLogger(__name__)

from config import config


# Lazily-bound in _get_clap() (kept as None at import time so torch / transformers
# / faiss are not loaded until the first CLAP-dependent tool runs). See _get_clap().
# from direct_renderer import DirectRenderer, cart2polar  # Removed due to flexoar dependency
# from ear_direct_renderer import EarDirectRenderer as DirectRenderer, cart2polar  # Removed - EarDirectRenderer not in ear_direct_renderer

# Import EAR Direct Renderer for spatial audio mixing
from ear_direct_renderer import EarDirectRenderer as DirectRenderer, cart2polar

# Import EAR components for rendering
try:
    from ear.core import bs2051, Renderer
    from ear.core.select_items import select_rendering_items
    # from ear.core.metadata_processing import preprocess_rendering_items  # Removed due to import issues
    from ear.core.objectbased.renderer import ObjectRenderer
    from ear.core.direct_speakers.renderer import DirectSpeakersRenderer
    # from ear.core.hoa.renderer import HoaRenderer  # Not available in this version
    from ear.core import geom
    from ear.common import CartesianPosition, PolarPosition
    logger.info("✓ EAR imports successful")
except ImportError as e:
    logger.warning(f"⚠ Warning: EAR imports failed (expected in some environments): {e}")

# HuggingFace database and CLAP models (optional, will be None if not available).
# These are loaded LAZILY on first tool use (see _get_clap) so that
# `import tools` stays fast and the MCP stdio handshake is not blocked by
# multi-second CLAP / transformers / faiss model loading.
hf_model, hf_tokenizer, hf_device, hf_index, hf_metadata = None, None, None, None, None
hf_audio_model, hf_processor = None, None
hf_text_index, hf_text_corpus = None, None
# Heavy ML dependencies, bound lazily by _get_clap() (kept None until first CLAP use):
torch = None
search_hf_audio = None
get_audio_from_metadata = None
_clap_loaded = False


def _get_clap() -> bool:
    """Lazily initialize the CLAP models, HF FAISS index and text corpus.

    Called at the start of each tool that needs them. The heavy loading runs
    once and the results are cached in the module-level globals. Returns True
    when the HF model set is available, False otherwise (in which case the
    globals stay None and the caller's existing ``is None`` guards produce the
    usual "not available" error message).
    """
    global hf_model, hf_tokenizer, hf_device, hf_index, hf_metadata
    global hf_audio_model, hf_processor
    global hf_text_index, hf_text_corpus
    global _clap_loaded
    global torch, search_hf_audio, get_audio_from_metadata

    if _clap_loaded:
        return hf_model is not None
    try:
        # Heavy ML stack (torch / transformers / faiss via database_hf) is imported
        # here, on first CLAP use, so `import tools` stays fast. database_hf is
        # imported first so its preload_nvidia_libs() runs before torch loads.
        import database_hf as _db
        import torch
        search_hf_audio = _db.search_hf_audio
        get_audio_from_metadata = _db.get_audio_from_metadata
        init_search = _db.init_search
        load_text_corpus_index = _db.load_text_corpus_index
        from transformers import ClapAudioModelWithProjection, ClapProcessor
        hf_model, hf_tokenizer, hf_device, hf_index, hf_metadata = init_search(
            index_file=config.database.hf_index_file,
            metadata_file=config.database.hf_metadata_file
        )
        # Also load audio model for mix analysis
        hf_audio_model = ClapAudioModelWithProjection.from_pretrained(config.database.clap_model_id)
        hf_processor = ClapProcessor.from_pretrained(config.database.clap_model_id)
        hf_audio_model.to(hf_device)
        # Load text corpus for labeling
        hf_text_index, hf_text_corpus = load_text_corpus_index(config.database.text_corpus_index)
        _clap_loaded = True
        logger.info("CLAP models + HF index loaded lazily on first use")
        return hf_model is not None
    except Exception as e:
        logger.warning(f"HF database / CLAP not available: {type(e).__name__}: {e}")
        return False


def read_file(filename: str) -> tuple[str, bool]:
    """Read the contents of a documentation or configuration file.
    
    Parameters:
    - filename: Name of the file to read (with extension, e.g., "SPATIAL_AUDIO_GUIDE.md")
    
    Returns:
    - tuple[str, bool]: (file contents, is_error)
    
    Common files:
    - SPATIAL_AUDIO_GUIDE.md: Coordinate system and speaker layouts
    - RENDERER_TOOL_GUIDE.md: Position format for render_spatial_mix
    """
    
    if not os.path.exists(filename):
        available_files = [f for f in os.listdir('.') if f.endswith(('.md', '.txt', '.json'))]
        if len(available_files) > 10:
            file_list = ', '.join(available_files[:10]) + f' ... ({len(available_files)} total files)'
        else:
            file_list = ', '.join(available_files)
        return (f'File "{filename}" not found. Available files: {file_list}. '
                f'Please check the filename and try again.'), True
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return f'Successfully read {filename}:\n\n{content}', False
    except Exception as e:
        return f'Error reading file "{filename}": {str(e)}', True


def mark_complete(output_file: str) -> tuple[str, bool]:
    """Mark the task as complete with the final output file.
    
    Call this when you have finished the task and created the final output.
    This signals that no further work is needed.
    
    Parameters:
    - output_file: The final output filename (without .wav extension)
    
    Returns:
    - tuple[str, bool]: (confirmation message, is_error)
    """
    output_file = strip_wav(output_file)
    if not os.path.exists(output_file + '.wav'):
        return f'File "{output_file}.wav" does not exist. Create the file first before marking complete.', True
    
    return f'Task marked complete. Final output: {output_file}.wav', False


def generate_single_sound(audio_type: str, dur: float, level_db: float=-20, suffix: str='', count: int=1, auto_trim: bool=False, trim_threshold: float=-30) -> tuple[str, bool]:
    """Generate a sound at a specified RMS level.
    
    Parameters:
    - audio_type: Description of the sound (e.g., "barking dog", "rain", "scream")
    - dur: Duration in seconds
    - level_db: Target RMS level in dB. Use values from analyze_mix_spatial output.
                Typical values: -30 (quiet), -20 (normal), -10 (loud)
    - suffix: Unique identifier for output filename
    - count: Number of variations to generate. Files named audiotype.wav, audiotype_2.wav, etc.
    - auto_trim: If True, remove quiet segments below trim_threshold
    - trim_threshold: Energy threshold in dB for auto_trim. Range: -60 (keep more quiet parts) to -10 (aggressive cutting of quiet parts). Default -30.
    
    Returns:
    - tuple[str, bool]: (result message, is_error)
    """

    # Lazily load CLAP / HF models on first use (keeps `import tools` fast).
    _get_clap()

    if count < 1:
        return 'Count must be at least 1.', True
    
    if level_db >= 0:
        return 'level_db must be negative. Typical values: -30 (quiet), -20 (normal), -10 (loud).', True
    
    # Check if HuggingFace database is available
    if hf_model is None:
        return 'HuggingFace audio database is not available. Please run database_hf.py to create the index first.', True
    
    # Search HuggingFace database — request enough results for all variations
    k = max(config.database.search_results, count)
    try:
        hf_results = search_hf_audio(audio_type, hf_model, hf_tokenizer, hf_device, hf_index, hf_metadata, k=k)
    except Exception as e:
        return f'Error searching HuggingFace database: {str(e)}', True
    
    if not hf_results:
        return f'No sounds found matching "{audio_type}". Try a different description.', True
    
    # Select matches, prefer different result for each variation
    weights = np.array([1 / r['distance'] for r in hf_results])
    probabilities = weights / np.sum(weights)
    replace = count > len(hf_results)
    matches = np.random.choice(hf_results, size=count, p=probabilities, replace=replace)
    
    # Generate file(s)
    results = []
    for i, match in enumerate(matches):
        # Determine filename
        if i == 0:
            ofn = audio_type + suffix
        else:
            ofn = audio_type + suffix + f'_{i + 1}'
        
        # Check overwrite
        ret = check_overwrite(ofn)
        if ret is not None:
            results.append(f'  {ofn}.wav: SKIPPED (file already exists)')
            continue
        
        # Get audio data from HuggingFace metadata (with auto_trim before looping)
        sig, fs = get_audio_from_metadata(match['metadata'], dur, auto_trim=auto_trim, trim_threshold_db=trim_threshold)
        
        # Normalize to target RMS level
        current_rms = np.sqrt(np.mean(sig ** 2))
        if current_rms > 0:
            current_db = 20 * np.log10(current_rms)
            gain_db = level_db - current_db
            sig = sig * 10 ** (gain_db / 20)
        
        # Prevent clipping (rare at typical level_db values)
        np.clip(sig, -0.99, 0.99, out=sig)
        
        sf.write(ofn + '.wav', sig, fs)
        results.append(f'  {ofn}.wav ({dur:.1f}s, {level_db} dB)')
    
    if count == 1:
        ofn = audio_type + suffix
        return f'Generated "{ofn}.wav" ({dur:.1f}s, {level_db} dB)', False
    else:
        return (f'Generated {len(results)} variation(s):\n' 
                + '\n'.join(results)), False


def sum_mixes(file_in: list, suffix: str='', gain: list=None) -> tuple[str, bool]:
    """Combine multiple multichannel mixes into one sum mix.
    
    Parameters:
    - file_in: List of multichannel mix filenames (without .wav extension)
    - suffix: Unique identifier for output filename
    - gain: List of gain values in dB, one per input file (e.g., [-3, 0]). Default: 0 for all
    
    Returns:
    - tuple[str, bool]: (result message, is_error)
    
    Note: Input files must be multichannel (stereo, surround, or immersive), not mono sounds.
    """

    file_in = [strip_wav(f) for f in file_in]
    nfiles = len(file_in)
    
    if nfiles < 2:
        return 'At least 2 input files are required. This tool is for combining existing mixes.', True
    
    if gain is None:
        gain = [0.0] * nfiles
    if len(gain) != nfiles:
        return f'Number of gain values ({len(gain)}) must match number of files ({nfiles}).', True
    
    nch = 2
    dur = 0
    fs = config.audio.sample_rate

    for fi in range(nfiles):
        ret = check_inputfile(file_in[fi] + '.wav')
        if ret is not None:
            return ret, True
        finfo = sf.info(file_in[fi] + '.wav')
        if finfo.channels < 2:
            return 'Input files should have minimum of 2 channels.', True
        nch = max(nch, finfo.channels)
        dur = max(dur, np.ceil(finfo.duration * fs).astype(int))

    s = np.zeros((dur, nch))
    for fi in range(nfiles):
        si, _ = sf.read(file_in[fi] + '.wav', dtype='float32')  # assume file is 48000
        s[:si.shape[0], :si.shape[1]] += 10**(gain[fi]/20) * si

    ofn = 'outmultichannelsum' + suffix
    ret = check_overwrite(ofn + '.wav')
    if ret is not None:
        return ret, True
    sf.write(ofn + '.wav', s * 0.9 / (np.max(s) + 1e-15), fs)
    
    output_duration = dur / fs
    return f'Created sum mix "{ofn}.wav" ({output_duration:.1f}s, {nch} channels) from {nfiles} input files.', False
    

def render_spatial_mix(file_in: list, mix_type: str, position: list, suffix: str='', fade_duration: float = 0.0) -> tuple[str, bool]:
    """Mix mono sounds with spatial positions and movement trajectories.
    
    Parameters:
    - file_in: List of mono audio filenames (without .wav extension)
    - mix_type: Output format - 'stereo', 'surround', or 'immersive'
    - position: List of positions, one per input file:
      - Static: [x, y, z] (3 numbers)
      - Delayed start: [time, x, y, z] (4 numbers)
      - Moving: [(t1, x1, y1, z1), (t2, x2, y2, z2), ...]
    - suffix: Unique identifier for output filename
    - fade_duration: Fade in/out duration in seconds (default: 0.0, no fade)
    
    Returns:
    - tuple[str, bool]: (result message, is_error)
    
    Coordinates: x=left/right (-1 to 1), y=front/back (-1 to 1), z=height (0 to 1)
    
    Example:
    - render_spatial_mix(['helicopter'], 'immersive', [[(0, -1, 0, 1), (5, 1, 0, 1)]])
    - render_spatial_mix(['sound1', 'sound2'], 'immersive', [[-1, 0, 0], [1, 0, 0]], fade_duration=0.1)
    """
    file_in = [strip_wav(f) for f in file_in]
    nfiles = len(file_in)
    if len(position) != nfiles:
        return f'Number of positions ({len(position)}) must match number of files ({nfiles}).', True
    
    # Normalize position format:
    #   [x, y, z]         -> [(0, x, y, z)]        (static at t=0)
    #   [t, x, y, z]      -> [(t, x, y, z)]        (static with delay)
    #   [(t,x,y,z), ...]  -> [(t,x,y,z), ...]      (trajectory, unchanged)
    normalized = []
    for fi, pos in enumerate(position):
        if not isinstance(pos, (list, tuple)) or len(pos) == 0:
            return (f'Position for file {file_in[fi]} must be [x, y, z], [time, x, y, z], '
                    f'or a list of keyframes [(time, x, y, z), ...].'), True
        
        # Detect format by inspecting the first element
        first = pos[0]
        
        if isinstance(first, (list, tuple)):
            # Trajectory: list of keyframes [(t, x, y, z), ...]
            # Handle string-encoded tuples: ['(0, -1, 0, 0)', '(5, 1, 0, 0)']
            if all(isinstance(p, str) for p in pos):
                try:
                    pos = [ast.literal_eval(p) for p in pos]
                except:
                    return (f'Could not parse keyframes for file {file_in[fi]}. '
                            f'Each keyframe must be (time, x, y, z) with numeric values.'), True
            
            # Validate each keyframe
            for kf in pos:
                if not isinstance(kf, (list, tuple)) or len(kf) != 4:
                    return (f'Each keyframe must be (time, x, y, z). '
                            f'Got: {kf} for file {file_in[fi]}.'), True
                if not all(isinstance(v, (int, float)) for v in kf):
                    return (f'All keyframe values must be numbers. '
                            f'Got: {kf} for file {file_in[fi]}.'), True
            
            normalized.append(list(pos))
        
        elif isinstance(first, (int, float)):
            # Flat format: [x, y, z] or [time, x, y, z]
            if not all(isinstance(v, (int, float)) for v in pos):
                return (f'Position for file {file_in[fi]} has mixed types. '
                        f'Use [x, y, z] or [time, x, y, z] with all numbers.'), True
            
            if len(pos) == 3:
                # [x, y, z] -> static at t=0
                x, y, z = pos
                if not all(-1 <= v <= 1 for v in (x, y, z)):
                    return f'Coordinates x, y, z must be between -1 and 1 for file {file_in[fi]}.', True
                normalized.append([(0, x, y, z)])
            elif len(pos) == 4:
                # [time, x, y, z] -> static with delay
                t, x, y, z = pos
                if t < 0:
                    return f'Time must be >= 0 for file {file_in[fi]}.', True
                if not all(-1 <= v <= 1 for v in (x, y, z)):
                    return f'Coordinates x, y, z must be between -1 and 1 for file {file_in[fi]}.', True
                normalized.append([(t, x, y, z)])
            else:
                return (f'Position for file {file_in[fi]} must have 3 values [x, y, z] '
                        f'or 4 values [time, x, y, z]. Got {len(pos)} values.'), True
        else:
            return (f'Could not parse position for file {file_in[fi]}. '
                    f'Use [x, y, z], [time, x, y, z], or [(time, x, y, z), ...].'), True
    
    position = normalized

    # Validate trajectories (time ordering, coordinate ranges)
    trajectory_spans = {}
    for fi, t in enumerate(position):
        current_time = 0
        for path in t:
            if path[0] < current_time:
                return 'Time values in trajectories should be >= 0 and increasing.', True
            if not all(-1 <= x <= 1 for x in path[1:]):
                return f'Coordinates x, y, z must be between -1 and 1 for file {file_in[fi]}.', True
            current_time = path[0]
        trajectory_spans[fi] = t[-1][0] - t[0][0]

    if mix_type not in ['stereo', 'surround', 'immersive']:
        return 'Mix type must be "stereo", "surround" or "immersive"', True
    if nfiles <= 0:
        return 'At least one input file is required.', True
    
    # Check input files
    for fi, filename in enumerate(file_in):
        ret = check_inputfile(filename + '.wav')
        if ret is not None:
            return ret, True
        if sf.info(filename + '.wav').channels != 1:
            return f'Input file {filename} has more than 1 channel. This tool is meant for mixing mono sounds. To combine multichannel mixes, use another tool.', True
        dur = sf.info(filename + '.wav').duration
        if dur < trajectory_spans[fi]:
            return f'Input file {filename} is only {dur} seconds long but the corresponding moving trajectory in parameter "position" extends beyond that. Use time values in parameter "position" that cover the sound duration or generate longer sounds.', True

    ofn = 'out' + mix_type + suffix
    ret = check_overwrite(ofn + '.wav')
    if ret is not None:
        return ret, True
    
    layout_name = config.layout.layout_map.get(mix_type.lower(), mix_type)
    gain = [1.0] * nfiles  # use as parameter if needed
    
    # Load and process audio files with optional fade
    processed_files = []
    fs = config.audio.sample_rate
    fade_samples = int(fade_duration * fs)
    
    for fi, filename in enumerate(file_in):
        try:
            sig, sig_fs = sf.read(filename + '.wav', dtype='float32')
            
            # Apply fade in/out if requested
            if fade_samples > 0 and len(sig) > fade_samples * 2:
                # Create fade envelope
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                sig[:fade_samples] *= fade_in
                sig[-fade_samples:] *= fade_out
            
            # Save processed file
            processed_name = f"_temp_{filename}"
            sf.write(processed_name + '.wav', sig, sig_fs)
            processed_files.append((filename, processed_name))
        except Exception as e:
            return f'Error processing file {filename}: {str(e)}', True
    
    try:
        renderer = DirectRenderer(layout_name, sample_rate=config.audio.sample_rate)
        
        for (orig_name, proc_name), traj, g in zip(processed_files, position, gain):
            filename = proc_name  # Use processed file
            # Validate trajectory format
            if not isinstance(traj, (list, tuple)) or len(traj) < 1:
                return f'Trajectory for file {filename} must have at least one keyframe.', True
            
            # Pass Cartesian coordinates directly to EAR panner (no polar conversion needed)
            adjusted_pos = []
            for keyframe in traj:
                if len(keyframe) != 4:
                    return f'Each keyframe must be (time, x, y, z). Got: {keyframe}', True
                t, x, y, z = keyframe
                # EAR panner expects Cartesian coordinates directly - no conversion needed
                adjusted_pos.append((t, x, y, z))

            # Moving trajectories are linearly interpolated inside
            # EarDirectRenderer.render(); no explicit densification is needed here.
            renderer.add_audio_object(filename + '.wav', adjusted_pos, gain=g)
        
        result = renderer.render(ofn + '.wav', normalize=True)
        
        # Clean up temporary files
        for orig_name, proc_name in processed_files:
            try:
                os.remove(proc_name + '.wav')
            except:
                pass
        
        return f'{result}', False
        
    except Exception as e:
        # Clean up temporary files on error
        for orig_name, proc_name in processed_files:
            try:
                os.remove(proc_name + '.wav')
            except:
                pass
        return f'Rendering failed: {str(e)}', True


# Spatial region definitions for common layouts
SPATIAL_REGIONS = {
    'stereo': {
        'left': [0],
        'right': [1],
    },
    'surround': {  # 5.1 (EAR: 0+5+0)
        'front_left': [0],
        'front_center': [2], 
        'front_right': [1],
        'surround_left': [4],
        'surround_right': [5],
        'lfe': [3]
    },
    'immersive': {  # 7.1.4 (EAR: 4+7+0)
        'front_left': [0],
        'front_right': [1],
        'front_center': [2],
        'side_left': [4],
        'side_right': [5],
        'rear_left': [6],
        'rear_right': [7],
        'height_front_left': [8],
        'height_front_right': [9],
        'height_rear_left': [10],
        'height_rear_right': [11],
    }
}




def analyze_mix_spatial(
    mix_file: str,
    segment_duration: float = 2.0,
    energy_threshold_db: float = -40
) -> tuple[str, bool]:
    """Analyze a multichannel mix to identify sounds, positions, timing, and loudness.
    
    Parameters:
    - mix_file: Path to multichannel audio file (without .wav extension)
    - segment_duration: Analysis window in seconds (default: 2.0)
    - energy_threshold_db: Minimum energy to analyze (default: -40)
    
    Returns:
    - tuple[str, bool]: (analysis result, is_error)
    """

    # Lazily load CLAP / HF models on first use (keeps `import tools` fast).
    _get_clap()

    mix_file = strip_wav(mix_file)
    if not os.path.exists(mix_file + '.wav'):
        return f'Mix file "{mix_file}.wav" not found.', True
    
    # Load audio
    audio, fs = sf.read(mix_file + '.wav', dtype='float32')
    if audio.ndim == 1:
        return 'Input must be a multichannel (stereo, surround, or immersive) mix file.', True
    
    duration = len(audio) / fs
    n_channels = audio.shape[1]
    
    # Detect layout type from channel count
    if n_channels == 2:
        layout = 'stereo'
    elif n_channels == 6:
        layout = 'surround'
    elif n_channels >= 12:
        layout = 'immersive'
    else:
        layout = 'surround'  # fallback
    
    regions = SPATIAL_REGIONS.get(layout, {})
    
    # Check CLAP audio model availability
    if hf_audio_model is None:
        return 'CLAP audio model not available. Cannot perform semantic analysis.', True
    
    # Check database availability
    if hf_index is None:
        return 'HuggingFace database not available. Cannot label sounds.', True
    
    # Segment analysis
    n_segments = int(np.ceil(duration / segment_duration))
    detected_sounds = []
    
    for seg_idx in range(n_segments):
        start_sample = int(seg_idx * segment_duration * fs)
        end_sample = int(min((seg_idx + 1) * segment_duration, duration) * fs)
        start_time = seg_idx * segment_duration
        end_time = min((seg_idx + 1) * segment_duration, duration)
        
        segment_audio = audio[start_sample:end_sample, :]
        
        # Analyze each spatial region
        for region_name, channel_indices in regions.items():
            # Get valid channels
            valid_channels = [c for c in channel_indices if c < n_channels]
            if not valid_channels:
                continue
            
            # Sum channels for this region
            region_audio = segment_audio[:, valid_channels].mean(axis=1)
            
            # Calculate energy
            rms = np.sqrt(np.mean(region_audio ** 2))
            energy_db = 20 * np.log10(rms + 1e-15)
            
            if energy_db < energy_threshold_db:
                continue
            
            # Generate CLAP embedding
            try:
                inputs = hf_processor(audio=region_audio, return_tensors="pt", sampling_rate=fs)
                inputs = {k: v.to(hf_device) for k, v in inputs.items()}
                with torch.no_grad():
                    embedding = hf_audio_model(**inputs).audio_embeds.cpu().numpy().squeeze()
            except Exception:
                continue
            
            # Get text labels using text corpus nearest neighbor
            labels = get_text_labels(embedding, hf_text_index, hf_text_corpus, k=5)
            primary_label = labels[0] if labels else 'unknown'
            
            detected_sounds.append({
                'segment': seg_idx + 1,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'region': region_name,
                'label': primary_label,
                'labels': labels,  # Top 5 labels
                'energy_db': float(energy_db),
                'embedding': embedding  # For internal similarity grouping (removed from output)
            })
    
    if not detected_sounds:
        return f'No significant sounds detected in "{mix_file}.wav" (energy below {energy_threshold_db} dB).', False
    
    # Group sounds with movement tracking
    grouped_sounds = group_sounds_with_movement(
        detected_sounds, 
        similarity_threshold=config.database.similarity_threshold
    )
    
    # Calculate total energy before limiting
    all_grouped_sounds = sorted(grouped_sounds, key=lambda s: s['energy_db'], reverse=True)
    total_energy = sum(10 ** (s['energy_db'] / 20) for s in all_grouped_sounds)
    
    # Sort by energy and take top N sounds
    grouped_sounds = all_grouped_sounds
    if config.database.max_analysis_sounds > 0:
        grouped_sounds = grouped_sounds[:config.database.max_analysis_sounds]
    
    # Calculate how much energy the prominent sounds capture
    prominent_energy = sum(10 ** (s['energy_db'] / 20) for s in grouped_sounds)
    coverage_pct = 100 * prominent_energy / total_energy if total_energy > 0 else 100
    omitted_count = len(all_grouped_sounds) - len(grouped_sounds)
    
    # Re-sort by start time for output
    grouped_sounds = sorted(grouped_sounds, key=lambda s: s['start_time'])
    
    # Calculate spatial balance
    region_energies = {}
    for region in regions.keys():
        # Check if any sound has this region in its regions list
        region_sounds = [s for s in grouped_sounds 
                         if s.get('regions') and s['regions'][0]['region'] == region]
        if region_sounds:
            region_energies[region] = max(s['energy_db'] for s in region_sounds)
        else:
            region_energies[region] = -60  # Very quiet
    
    # Categorize regions
    front_regions = ['front_left', 'front_center', 'front_right', 'left', 'right']
    surround_regions = ['side_left', 'side_right', 'rear_left', 'rear_right', 'surround_left', 'surround_right']
    height_regions = ['height_front_left', 'height_front_right', 'height_rear_left', 'height_rear_right']
    
    front_energy = max([region_energies.get(r, -60) for r in front_regions])
    surround_energy = max([region_energies.get(r, -60) for r in surround_regions]) if surround_regions else -60
    height_energy = max([region_energies.get(r, -60) for r in height_regions]) if height_regions else -60
    
    total_energy = 10 ** (front_energy/20) + 10 ** (surround_energy/20) + 10 ** (height_energy/20) + 1e-10
    
    front_pct = 100 * 10 ** (front_energy/20) / total_energy
    surround_pct = 100 * 10 ** (surround_energy/20) / total_energy
    height_pct = 100 * 10 ** (height_energy/20) / total_energy
    
    # Analyze temporal characteristics
    continuity_score, transition_type, dynamic_range_db = analyze_temporal_continuity(audio, fs)
    
    # Analyze content density and gaps
    # Create timeline of content presence
    segment_has_content = {}
    for seg_idx in range(n_segments):
        start_time = seg_idx * segment_duration
        end_time = min((seg_idx + 1) * segment_duration, duration)
        segment_has_content[(start_time, end_time)] = False
    
    for sound in detected_sounds:
        for (start, end) in segment_has_content.keys():
            if sound['start_time'] < end and sound['end_time'] > start:
                segment_has_content[(start, end)] = True
    
    # Find silent segments (gaps)
    silent_segments = []
    current_silent_start = None
    for (start, end), has_content in sorted(segment_has_content.items()):
        if not has_content:
            if current_silent_start is None:
                current_silent_start = start
        else:
            if current_silent_start is not None:
                silent_segments.append((current_silent_start, start))
                current_silent_start = None
    # Handle trailing silence
    if current_silent_start is not None:
        silent_segments.append((current_silent_start, duration))
    
    # Calculate density by region (beginning, middle, end)
    third = duration / 3
    beginning_sounds = len([s for s in detected_sounds if s['start_time'] < third])
    middle_sounds = len([s for s in detected_sounds if third <= s['start_time'] < 2 * third])
    end_sounds = len([s for s in detected_sounds if s['start_time'] >= 2 * third])
    
    # Format output
    lines = []
    lines.append(f"Mix Analysis: {mix_file}.wav ({layout}, {duration:.1f}s)")
    lines.append("")
    lines.append("SPATIAL BALANCE:")
    lines.append(f"  Front channels: {front_pct:.0f}% of energy")
    if surround_regions:
        lines.append(f"  Surround channels: {surround_pct:.0f}% of energy")
    if height_regions:
        lines.append(f"  Height channels: {height_pct:.0f}% of energy")
    lines.append("")
    lines.append("TEMPORAL CHARACTERISTICS:")
    lines.append(f"  Dynamic range: {dynamic_range_db:.0f} dB")
    lines.append(f"  Transition style: {transition_type}")
    lines.append(f"  Continuity score: {continuity_score:.2f}")
    lines.append("")
    lines.append("CONTENT DENSITY:")
    lines.append(f"  Beginning (0-{third:.1f}s): {beginning_sounds} sounds")
    lines.append(f"  Middle ({third:.1f}-{2*third:.1f}s): {middle_sounds} sounds")
    lines.append(f"  End ({2*third:.1f}-{duration:.1f}s): {end_sounds} sounds")
    total_silent = sum(e - s for s, e in silent_segments)
    lines.append(f"  Total silent: {total_silent:.1f}s")
    if silent_segments:
        lines.append(f"  Silent gaps: {len(silent_segments)}")
        for start, end in silent_segments[:3]:  # Show up to 3
            lines.append(f"    - {start:.1f}-{end:.1f}s")
        if len(silent_segments) > 3:
            lines.append(f"    - ... and {len(silent_segments) - 3} more")
    lines.append("")
    lines.append("CHANNEL ACTIVITY:")
    # Categorize channels by activity level
    active_channels = []
    quiet_channels = []
    silent_channels = []
    for region in regions.keys():
        energy = region_energies.get(region, -60)
        if energy > -30:
            active_channels.append(region)
        elif energy > -50:
            quiet_channels.append(region)
        else:
            silent_channels.append(region)
    
    if active_channels:
        lines.append(f"  Active: {', '.join(active_channels)}")
    if quiet_channels:
        lines.append(f"  Quiet: {', '.join(quiet_channels)}")
    if silent_channels:
        lines.append(f"  Silent: {', '.join(silent_channels)}")
    
    lines.append("")
    lines.append("PROMINENT SOUNDS (top {} by energy):".format(config.database.max_analysis_sounds))
    lines.append(f"  Coverage: {coverage_pct:.0f}% of detected sound energy")
    if omitted_count > 0:
        lines.append(f"  Omitted: {omitted_count} quieter sound(s) not shown")
    
    # Sort by start time
    for sound in sorted(grouped_sounds, key=lambda s: s['start_time']):
        start_time = sound['start_time']
        end_time = sound['end_time']
        time_str = f"{start_time:.1f}-{end_time:.1f}s"
        duration = end_time - start_time
        loudness = "loud" if sound['energy_db'] > -10 else "moderate" if sound['energy_db'] > -30 else "quiet"
        # Show top 3 labels
        labels_str = ', '.join(sound.get('labels', [sound['label']])[:3])
        # Simplify trajectory: only show start and end positions
        sound_regions = sound.get('regions', [])
        if len(sound_regions) > 1:
            # Moving sound: show start → end only (linear trajectory)
            start_region = sound_regions[0]['region']
            end_region = sound_regions[-1]['region']
            if start_region == end_region:
                # Same region, treat as stationary
                lines.append(f"  [{time_str}] \"{labels_str}\" at {start_region}, stationary, {loudness}, duration: {duration:.1f}s, start: {start_time:.1f}s")
            else:
                # Different regions, show linear movement
                lines.append(f"  [{time_str}] \"{labels_str}\" moving: {start_region} → {end_region}, {loudness}, duration: {duration:.1f}s, start: {start_time:.1f}s")
        elif sound_regions:
            # Single region: stationary
            lines.append(f"  [{time_str}] \"{labels_str}\" at {sound_regions[0]['region']}, stationary, {loudness}, duration: {duration:.1f}s, start: {start_time:.1f}s")
        else:
            lines.append(f"  [{time_str}] \"{labels_str}\", {loudness}, duration: {duration:.1f}s, start: {start_time:.1f}s")
    
    return '\n'.join(lines), False


def mimic_mix_direct(
    mix_file: str,
    segment_duration: float = 2.0,
    energy_threshold_db: float = -40,
    output_suffix: str = '_mimic'
) -> tuple[str, bool]:
    """Mimic a multichannel mix by directly constructing channel signals.
    
    This approach preserves the original channel distribution by:
    1. Analyzing each channel separately with overlapping segments
    2. Using embedding distance to determine if sounds are the same (panned) or different (diffuse)
    3. Tracking sound continuity across segments to reuse waveforms
    4. Constructing output channels with crossfading for smooth transitions
    5. Adapting overlap and crossfade to match source temporal characteristics
    
    Parameters:
    - mix_file: Path to multichannel audio file (without .wav extension)
    - segment_duration: Analysis window in seconds (default: 2.0)
    - energy_threshold_db: Minimum energy to analyze (default: -40)
    - output_suffix: Suffix for output file (default: '_mimic')
    
    Returns:
    - tuple[str, bool]: (result message, is_error)
    """
    mix_file = strip_wav(mix_file)
    if not os.path.exists(mix_file + '.wav'):
        return f'Mix file "{mix_file}.wav" not found.', True
    
    # Lazily load CLAP / HF models on first use (keeps `import tools` fast).
    _get_clap()

    # Check dependencies
    if hf_audio_model is None:
        return 'CLAP audio model not available. Cannot perform analysis.', True
    if hf_index is None:
        return 'HuggingFace database not available. Cannot mimic mix.', True
    
    # Load audio
    audio, fs = sf.read(mix_file + '.wav', dtype='float32')
    if audio.ndim == 1:
        return 'Input must be a multichannel (stereo, surround, or immersive) mix file.', True
    
    duration = len(audio) / fs
    n_channels = audio.shape[1]
    
    # Detect layout type from channel count
    if n_channels == 2:
        layout = 'stereo'
    elif n_channels == 6:
        layout = 'surround'
    elif n_channels >= 12:
        layout = 'immersive'
    else:
        layout = 'surround'
    
    regions = SPATIAL_REGIONS.get(layout, {})
    channel_to_region = {}
    for region_name, channel_indices in regions.items():
        for ch in channel_indices:
            if ch < n_channels:
                channel_to_region[ch] = region_name
    
    # Analyze temporal continuity of source to determine parameters
    source_continuity, source_transition, source_dynamic_range = analyze_temporal_continuity(audio, fs)
    
    # Set overlap and crossfade based on source characteristics
    if source_continuity > 0.7:  # Smooth source
        overlap_ratio = 0.75
        crossfade_duration = 0.3
    elif source_continuity < 0.3:  # Choppy source
        overlap_ratio = 0.3
        crossfade_duration = 0.05
    else:  # Mixed
        overlap_ratio = 0.5
        crossfade_duration = 0.2
    
    # Calculate segment hop based on overlap
    overlap_ratio = min(0.75, max(0, overlap_ratio))  # Clamp to 0-0.75
    segment_hop = segment_duration * (1 - overlap_ratio)
    crossfade_samples = int(crossfade_duration * fs)
    
    # Analyze with overlapping segments
    segment_data = []
    seg_idx = 0
    current_time = 0.0
    
    while current_time < duration:
        start_time = current_time
        end_time = min(current_time + segment_duration, duration)
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        segment_audio = audio[start_sample:end_sample, :]
        
        channel_info = {}
        for ch in range(n_channels):
            if end_sample - start_sample < 100:  # Skip very short segments
                continue
            ch_audio = segment_audio[:, ch]
            
            # Calculate energy
            rms = np.sqrt(np.mean(ch_audio ** 2))
            energy_db = 20 * np.log10(rms + 1e-15)
            
            if energy_db < energy_threshold_db:
                continue
            
            # Generate embedding
            try:
                inputs = hf_processor(audio=ch_audio, return_tensors="pt", sampling_rate=fs)
                inputs = {k: v.to(hf_device) for k, v in inputs.items()}
                with torch.no_grad():
                    embedding = hf_audio_model(**inputs).audio_embeds.cpu().numpy().squeeze()
            except Exception:
                continue
            
            # Get label
            labels = get_text_labels(embedding, hf_text_index, hf_text_corpus, k=3)
            
            channel_info[ch] = {
                'embedding': embedding,
                'energy_db': energy_db,
                'label': labels[0] if labels else 'unknown',
                'labels': labels
            }
        
        if channel_info:
            segment_data.append({
                'segment_idx': seg_idx,
                'start_time': start_time,
                'end_time': end_time,
                'channels': channel_info
            })
        
        seg_idx += 1
        current_time += segment_hop
    
    if not segment_data:
        return f'No significant sounds detected in "{mix_file}.wav" (energy below {energy_threshold_db} dB).', True
    
    # Group sounds across channels by embedding similarity (using config threshold)
    sound_groups = group_sounds_by_embedding(segment_data, config.database.similarity_threshold)
    
    # Sort groups by start time
    sorted_groups = sorted(sound_groups, key=lambda g: g['start_time'])
    
    # First pass: calculate total duration for each continuous sound
    # This ensures we generate sounds with the correct length
    # Use embedding similarity instead of label matching to correctly identify same sounds
    sound_durations = {}  # key -> {'start', 'end', 'channels', 'is_panned', 'embedding', 'label', 'groups'}
    
    for group in sorted_groups:
        # Get reference embedding from highest energy channel
        ref_channel = max(group['channels'].items(), key=lambda x: x[1]['energy_db'])[0]
        ref_info = group['channels'][ref_channel]
        label = ref_info['label']
        embedding = ref_info['embedding']
        is_panned = group['is_panned']
        
        # Find existing sound with similar embedding
        found_key = None
        for key, dur_info in sound_durations.items():
            # Check if time is continuous
            prev_end = dur_info['end']
            if group['start_time'] > prev_end + segment_hop * 1.5:
                continue  # Too far apart in time
            
            # Check embedding similarity
            emb_distance = cosine_distance(np.array(dur_info['embedding']), np.array(embedding))
            if emb_distance < config.database.similarity_threshold:
                found_key = key
                break
        
        if found_key is not None:
            # Extend existing sound
            sound_durations[found_key]['end'] = max(sound_durations[found_key]['end'], group['end_time'])
            sound_durations[found_key]['channels'].update(group['channels'].keys())
            sound_durations[found_key]['groups'].append(group)
        else:
            # Create new entry with unique key
            key = f"{label}_{group['start_time']:.1f}"
            sound_durations[key] = {
                'start': group['start_time'],
                'end': group['end_time'],
                'channels': set(group['channels'].keys()),
                'is_panned': is_panned,
                'embedding': embedding,
                'label': label,
                'groups': [group]
            }
    
    # Track which sounds have been generated
    generated_labels = set()
    
    # Generate waveforms and construct output
    output_audio = np.zeros((int(duration * fs), n_channels), dtype='float32')
    generated_sounds = []
    
    for group in sorted_groups:
        is_panned = group['is_panned']
        group_channels = frozenset(group['channels'].keys())
        
        # Find the label and check if already generated
        if is_panned:
            ref_channel = max(group['channels'].items(), key=lambda x: x[1]['energy_db'])[0]
            ref_info = group['channels'][ref_channel]
            label = ref_info['label']
        else:
            # For diffuse, use first channel's label
            label = list(group['channels'].values())[0]['label']
        
        # Find the sound duration entry
        duration_key = None
        for key, dur_info in sound_durations.items():
            if key.startswith(label) or key == label:
                if group['start_time'] >= dur_info['start'] - 0.1 and group['start_time'] <= dur_info['end'] + 0.1:
                    duration_key = key
                    break
        
        if duration_key is None:
            continue
        
        # Check if this sound was already generated
        if duration_key in generated_labels:
            continue
        
        # Mark as generated
        generated_labels.add(duration_key)
        
        # Get total duration info
        dur_info = sound_durations[duration_key]
        total_start = dur_info['start']
        total_end = dur_info['end']
        total_dur = total_end - total_start
        
        if is_panned:
            # PANNED: Same waveform, different gains per channel
            # Use the group with highest energy for reference
            ref_channel = max(group['channels'].items(), key=lambda x: x[1]['energy_db'])[0]
            ref_info = group['channels'][ref_channel]
            
            # Search database for sound
            embedding = ref_info['embedding'].reshape(1, -1).astype('float32')
            distances, indices = hf_index.search(embedding, 5)
            
            valid_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) 
                              if idx < len(hf_metadata)]
            if not valid_results:
                continue
            
            weights = [1.0 / (d + 0.01) for _, d in valid_results]
            selected_idx = random.choices(range(len(valid_results)), weights=weights, k=1)[0]
            best_idx = valid_results[selected_idx][0]
            
            # Get audio with TOTAL duration
            try:
                sig, sound_fs = get_audio_from_metadata(hf_metadata[best_idx], total_dur)
            except Exception:
                continue
            
            # Measure source RMS and calculate gain to match target energy_db
            source_rms = np.sqrt(np.mean(sig ** 2))
            if source_rms > 0:
                source_db = 20 * np.log10(source_rms)
            else:
                source_db = -60
            
            # Apply to each channel with RMS-matched gain
            n_samples = len(sig)
            start_sample = int(total_start * fs)
            end_sample = min(start_sample + n_samples, int(duration * fs))
            actual_samples = end_sample - start_sample
            
            for ch, ch_info in group['channels'].items():
                # Calculate gain to match target energy_db
                target_db = ch_info['energy_db']
                gain_db = target_db - source_db
                gain_linear = 10 ** (gain_db / 20)
                audio_segment = sig[:actual_samples] * gain_linear
                
                # Apply crossfade envelope
                if crossfade_samples > 0 and actual_samples > crossfade_samples * 2:
                    fade_in = np.linspace(0, 1, min(crossfade_samples, actual_samples // 2))
                    audio_segment[:len(fade_in)] *= fade_in
                    fade_out = np.linspace(1, 0, min(crossfade_samples, actual_samples // 2))
                    audio_segment[-len(fade_out):] *= fade_out
                
                output_audio[start_sample:end_sample, ch] += audio_segment
            
            generated_sounds.append({
                'label': label,
                'channels': list(group_channels),
                'type': 'panned',
                'time': f"{total_start:.1f}-{total_end:.1f}s ({total_dur:.1f}s)"
            })
        
        else:
            # DIFFUSE: Different waveforms per channel
            for ch, ch_info in group['channels'].items():
                ch_label = ch_info['label']
                
                # Search database for sound
                embedding = ch_info['embedding'].reshape(1, -1).astype('float32')
                distances, indices = hf_index.search(embedding, 5)
                
                valid_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) 
                                  if idx < len(hf_metadata)]
                if not valid_results:
                    continue
                
                weights = [1.0 / (d + 0.01) for _, d in valid_results]
                selected_idx = random.choices(range(len(valid_results)), weights=weights, k=1)[0]
                best_idx = valid_results[selected_idx][0]
                
                # Get audio with TOTAL duration
                try:
                    sig, sound_fs = get_audio_from_metadata(hf_metadata[best_idx], total_dur)
                except Exception:
                    continue
                
                # Measure source RMS and calculate gain to match target energy_db
                source_rms = np.sqrt(np.mean(sig ** 2))
                if source_rms > 0:
                    source_db = 20 * np.log10(source_rms)
                else:
                    source_db = -60
                
                # Calculate gain to match target energy_db
                target_db = ch_info['energy_db']
                gain_db = target_db - source_db
                gain_linear = 10 ** (gain_db / 20)
                
                n_samples = len(sig)
                start_sample = int(total_start * fs)
                end_sample = min(start_sample + n_samples, int(duration * fs))
                actual_samples = end_sample - start_sample
                
                audio_segment = sig[:actual_samples] * gain_linear
                
                # Apply crossfade envelope
                if crossfade_samples > 0 and actual_samples > crossfade_samples * 2:
                    fade_in = np.linspace(0, 1, min(crossfade_samples, actual_samples // 2))
                    audio_segment[:len(fade_in)] *= fade_in
                    fade_out = np.linspace(1, 0, min(crossfade_samples, actual_samples // 2))
                    audio_segment[-len(fade_out):] *= fade_out
                
                output_audio[start_sample:end_sample, ch] += audio_segment
            
            generated_sounds.append({
                'label': label,
                'channels': list(group['channels'].keys()),
                'type': 'diffuse',
                'time': f"{total_start:.1f}-{total_end:.1f}s ({total_dur:.1f}s)"
            })
    
    # Apply global energy envelope from original mix
    output_audio = apply_energy_envelope(output_audio, audio, fs)
    
    # Normalize output
    peak = np.max(np.abs(output_audio))
    if peak > 0:
        output_audio = output_audio * 0.9 / peak
    
    # Save output
    output_file = mix_file + output_suffix + '.wav'
    sf.write(output_file, output_audio, fs)
    
    # Format output message
    lines = []
    lines.append(f"Direct Mimic: {mix_file}.wav → {output_file}")
    lines.append(f"Layout: {layout} ({n_channels} channels), Duration: {duration:.1f}s")
    lines.append(f"Temporal: {source_transition} transitions, {source_dynamic_range:.0f} dB range")
    lines.append(f"Parameters: {overlap_ratio*100:.0f}% overlap, {crossfade_duration:.2f}s crossfade")
    lines.append("")
    lines.append("GENERATED SOUNDS:")
    
    for sound in generated_sounds:
        channel_names = [channel_to_region.get(ch, f'ch{ch}') for ch in sound['channels']]
        lines.append(f"  [{sound['time']}] \"{sound['label']}\" ({sound['type']}) in: {', '.join(channel_names)}")
    
    lines.append("")
    lines.append(f"OUTPUT: Created mimic mix \"{output_file}\" with {len(generated_sounds)} sound groups")
    
    return '\n'.join(lines), False




