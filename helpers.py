import glob
import os
import numpy as np
import ctypes


def preload_nvidia_libs():
    """Preload missing NVIDIA runtime libraries required by TorchCodec/FFmpeg before importing PyTorch dependencies."""
    if os.environ.get("CONDA_PREFIX"):
        _nvidia_libs = glob.glob(f"{os.environ['CONDA_PREFIX']}/lib/python3.*/site-packages/nvidia/*/lib/*.so.12")
        for _lib in _nvidia_libs:
            try:
                ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass  # Fallback gracefully if libraries are globally installed elsewhere


def check_overwrite(fn):
    if os.path.exists(fn):
        # Auto-suggest a unique filename by appending a number
        base, ext = os.path.splitext(fn)
        n = 2
        while os.path.exists(f'{base}_{n}{ext}'):
            n += 1
        suggested = f'{base}_{n}{ext}'
        return f'File "{fn}" already exists. Use a different suffix to avoid overwriting. Suggested alternative: "{os.path.splitext(suggested)[0]}" (the suffix would be "_{n}").'
    else:
        pass
    
    
def check_inputfile(fn):
    if not os.path.exists(fn):
        filenames_without_ext = [os.path.splitext(f)[0] for f in glob.glob('*.wav')]
        # Limit listing to 15 files to avoid very long error messages
        if len(filenames_without_ext) > 15:
            file_list = ', '.join(filenames_without_ext[:15]) + f' ... ({len(filenames_without_ext)} total files)'
        else:
            file_list = ', '.join(filenames_without_ext)
        return (f'Input audio file "{os.path.splitext(fn)[0]}" does not exist. '
                f'Available files: [{file_list}]. '
                f'Use the correct file name when calling this function, or generate the audio first.')
    else:
        pass


def densify_pos(positions, step=0.1):
    """
    Perform linear interpolation between two 3D coordinates over a set of intermediate times.
    """

    assert step > 0, "Step should be greater than zero."
    assert len(positions) >= 2, "At least two coordinate sets are required for interpolation."
    
    # Validate timestamps are strictly increasing
    for i in range(len(positions) - 1):
        t1 = positions[i][0]
        t2 = positions[i + 1][0]
        assert t2 > t1, f"Timestamps must be strictly increasing. Got t[{i}]={t1}, t[{i+1}]={t2}"
    
    all_segments = []

    for i in range(len(positions) - 1):
        t1, x1, y1, z1 = positions[i]
        t2, x2, y2, z2 = positions[i + 1]
        num_points = int((t2 - t1) / step) + 1
        times = np.linspace(t1, t2, num_points)

        # Normalize the times to the range [0, 1]
        normalized_times = (times - t1) / (t2 - t1)

        # Perform linear interpolation for x, y, z
        x_interp = x1 + (x2 - x1) * normalized_times
        y_interp = y1 + (y2 - y1) * normalized_times
        z_interp = z1 + (z2 - z1) * normalized_times

        # Combine the interpolated coordinates
        segment = np.column_stack((times, x_interp, y_interp, z_interp))

        # Avoid duplicating the last point of a segment (it's the first of the next)
        if i < len(positions) - 2:
            segment = segment[:-1]

        all_segments.append(segment)

    interpolated_coords = [tuple(row) for row in np.vstack(all_segments).tolist()]

    return interpolated_coords


def strip_wav(name):
    """Strip .wav extension if accidentally included by the LLM."""
    if isinstance(name, str):
        return name.removesuffix('.wav')
    return name


def get_text_labels(embedding, text_index, text_corpus, k=5):
    """Get text labels for an audio embedding using text corpus nearest neighbor."""
    if text_index is None or text_corpus is None:
        return ['unknown']
    
    embedding = embedding.reshape(1, -1).astype('float32')
    distances, indices = text_index.search(embedding, k)
    
    labels = []
    for i, idx in enumerate(indices[0]):
        if idx < len(text_corpus):
            labels.append(text_corpus[idx])
    
    return labels if labels else ['unknown']


def cosine_distance(emb1, emb2):
    """Calculate cosine distance between two embeddings.
    
    Cosine distance = 1 - cosine_similarity
    Range: 0 (identical) to 2 (opposite)
    Typical: 0 (same) to 1 (orthogonal) for most audio embeddings
    
    Parameters:
    - emb1, emb2: Embedding vectors
    
    Returns:
    - Cosine distance (lower = more similar)
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors
    cosine_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    # Clamp to [-1, 1] to handle numerical errors
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    return 1.0 - cosine_sim


def group_sounds_with_movement(detected_sounds, min_movement_duration=1.0, similarity_threshold=None):
    """Group detected sounds, tracking region changes for movement detection.
    
    Parameters:
    - detected_sounds: List of detected sound segments
    - min_movement_duration: Minimum duration in a region to count as movement (default: 1.0s)
    - similarity_threshold: If provided, combine sounds with cosine distance < threshold.
                            Range: 0 (identical) to 2 (opposite). 
                            Typical: 0.1 (strict) to 0.5 (relaxed). None = use label match only.
    
    Returns:
    - List of grouped sounds, each with 'regions' list for movement tracking
    """
    if not detected_sounds:
        return []
    
    # Sort by start time
    sorted_sounds = sorted(detected_sounds, key=lambda s: s['start_time'])
    
    grouped = []
    
    for sound in sorted_sounds:
        label = sound['label']
        region = sound['region']
        start = sound['start_time']
        end = sound['end_time']
        energy = sound['energy_db']
        embedding = sound.get('embedding')
        labels = sound.get('labels', [label])
        
        # Find existing group to combine with
        found = False
        for group in grouped:
            # Check if this group overlaps in time
            if group['end_time'] < start - 0.5:
                continue
            
            # Determine if sounds should be combined
            should_combine = False
            
            if similarity_threshold is not None and embedding is not None and group.get('embedding') is not None:
                # Use cosine distance
                emb_distance = cosine_distance(np.array(group['embedding']), np.array(embedding))
                should_combine = emb_distance < similarity_threshold
            else:
                # Fall back to label match
                should_combine = group['label'] == label
            
            if should_combine:
                # Check if this continues the current region or starts a new one
                current_region = group['regions'][-1]
                
                if current_region['region'] == region:
                    # Same region: extend duration
                    current_region['end_time'] = end
                    current_region['energy_db'] = max(current_region['energy_db'], energy)
                else:
                    # Different region: check if this is a movement
                    duration_in_current = current_region['end_time'] - current_region['start_time']
                    if duration_in_current >= min_movement_duration:
                        # Significant time in previous region, this is movement
                        group['regions'].append({
                            'region': region,
                            'start_time': start,
                            'end_time': end,
                            'energy_db': energy
                        })
                    else:
                        # Too brief, just update the region
                        current_region['region'] = region
                        current_region['end_time'] = end
                
                # Update group end time
                group['end_time'] = max(group['end_time'], end)
                group['energy_db'] = max(group['energy_db'], energy)
                
                # Combine labels
                existing_labels = group.get('all_labels', [group['label']])
                for lbl in labels:
                    if lbl not in existing_labels:
                        existing_labels.append(lbl)
                group['all_labels'] = existing_labels
                group['label'] = ', '.join(existing_labels[:3])  # Top 3 labels
                
                # Keep best embedding
                if energy > group.get('max_energy_db', -100):
                    group['max_energy_db'] = energy
                    group['embedding'] = embedding
                
                found = True
                break
        
        if not found:
            # Create new group
            grouped.append({
                'label': label,
                'labels': labels,
                'all_labels': labels,  # Track all labels for combining
                'start_time': start,
                'end_time': end,
                'energy_db': energy,
                'max_energy_db': energy,
                'embedding': embedding,
                'regions': [{
                    'region': region,
                    'start_time': start,
                    'end_time': end,
                    'energy_db': energy
                }]
            })
    
    # Second pass: merge groups with same time range AND similar embedding (same sound in multiple channels)
    # Note: embedding is still available at this point (removed after this pass)
    if similarity_threshold is not None:
        merged = []
        for group in grouped:
            found_merge = False
            for existing in merged:
                # Check if time ranges overlap significantly
                time_overlap = min(group['end_time'], existing['end_time']) - max(group['start_time'], existing['start_time'])
                min_duration = min(group['end_time'] - group['start_time'], existing['end_time'] - existing['start_time'])
                
                # Also check embedding similarity - only merge if same sound in different channels
                emb_distance = cosine_distance(np.array(existing['embedding']), np.array(group['embedding']))
                
                if time_overlap > min_duration * 0.5 and emb_distance < similarity_threshold:
                    # Merge regions from this group into existing
                    for region in group['regions']:
                        # Check if this region already exists
                        region_exists = False
                        for ex_region in existing['regions']:
                            if ex_region['region'] == region['region']:
                                # Extend existing region
                                ex_region['start_time'] = min(ex_region['start_time'], region['start_time'])
                                ex_region['end_time'] = max(ex_region['end_time'], region['end_time'])
                                region_exists = True
                                break
                        if not region_exists:
                            existing['regions'].append(region)
                    # Update time range
                    existing['start_time'] = min(existing['start_time'], group['start_time'])
                    existing['end_time'] = max(existing['end_time'], group['end_time'])
                    # Combine labels
                    existing_labels = existing.get('labels', [existing['label']])
                    group_labels = group.get('labels', [group['label']])
                    combined = list(dict.fromkeys(existing_labels + group_labels))  # Preserve order, remove duplicates
                    existing['labels'] = combined[:5]  # Keep top 5
                    existing['label'] = ', '.join(combined[:3])
                    found_merge = True
                    break
            if not found_merge:
                merged.append(group)
        grouped = merged
    
    # Clean up internal tracking fields (after second pass)
    for group in grouped:
        group.pop('max_energy_db', None)
        group.pop('all_labels', None)
        group.pop('embedding', None)  # Remove embedding from output
    
    return grouped


def group_sounds_by_embedding(segment_data, distance_threshold):
    """Group sounds across channels by embedding similarity.
    
    For each time segment, groups channels with similar embeddings as the same sound (panned).
    Channels with different embeddings are treated as different sounds (diffuse).
    
    Parameters:
    - segment_data: List of segment dicts, each with 'channels' dict containing embedding info
    - distance_threshold: Embedding distance threshold for grouping. If None, uses label matching.
    
    Returns:
    - List of sound groups, each with:
        - 'channels': dict of channel -> {embedding, energy_db, label}
        - 'is_panned': bool (True if same waveform should be used)
        - 'start_time', 'end_time': time range
    """
    import numpy as np
    
    sound_groups = []
    
    for segment in segment_data:
        start_time = segment['start_time']
        end_time = segment['end_time']
        channels = segment['channels']
        
        if not channels:
            continue
        
        # Cluster channels by embedding similarity or label match
        channel_list = list(channels.keys())
        visited = set()
        
        for ch1 in channel_list:
            if ch1 in visited:
                continue
            
            group_channels = {ch1: channels[ch1]}
            visited.add(ch1)
            
            for ch2 in channel_list:
                if ch2 in visited:
                    continue
                
                # Determine if channels should be grouped
                should_group = False
                if distance_threshold is not None:
                    # Use cosine distance
                    emb1 = channels[ch1]['embedding']
                    emb2 = channels[ch2]['embedding']
                    distance = cosine_distance(emb1, emb2)
                    should_group = distance < distance_threshold
                else:
                    # Fall back to label match
                    should_group = channels[ch1]['label'] == channels[ch2]['label']
                
                if should_group:
                    group_channels[ch2] = channels[ch2]
                    visited.add(ch2)
            
            # Determine if panned or diffuse
            if len(group_channels) == 1:
                is_panned = True
            elif distance_threshold is not None:
                max_distance = 0
                for ch_a in group_channels:
                    for ch_b in group_channels:
                        if ch_a < ch_b:
                            dist = cosine_distance(
                                group_channels[ch_a]['embedding'], 
                                group_channels[ch_b]['embedding']
                            )
                            max_distance = max(max_distance, dist)
                is_panned = max_distance < distance_threshold
            else:
                # With label match, assume panned if same label
                is_panned = True
            
            sound_groups.append({
                'channels': group_channels,
                'is_panned': is_panned,
                'start_time': start_time,
                'end_time': end_time
            })
    
    return sound_groups


def compute_energy_envelope(audio, fs, window=0.5):
    """Compute energy envelope of audio.
    
    Parameters:
    - audio: Audio array (samples x channels)
    - fs: Sample rate
    - window: Window size in seconds
    
    Returns:
    - Envelope array (one value per window)
    """
    window_samples = int(window * fs)
    n_windows = len(audio) // window_samples
    
    if n_windows == 0:
        return np.array([np.sqrt(np.mean(audio ** 2))])
    
    envelope = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        segment = audio[start:end]
        rms = np.sqrt(np.mean(segment ** 2))
        envelope.append(rms)
    
    return np.array(envelope)


def apply_energy_envelope(target_audio, reference_audio, fs, window=0.5, smoothing=0.5):
    """Apply reference audio's energy envelope to target.
    
    This preserves the overall dynamic shape of the reference audio -
    when it builds, peaks, and releases.
    
    Parameters:
    - target_audio: Audio to modify
    - reference_audio: Reference audio with desired envelope
    - fs: Sample rate
    - window: Analysis window in seconds
    - smoothing: Smoothing factor for gain curve (0 = no smoothing, 1 = very smooth)
    
    Returns:
    - Modified target audio
    """
    from scipy.interpolate import interp1d
    
    # Compute energy envelopes
    ref_envelope = compute_energy_envelope(reference_audio, fs, window)
    target_envelope = compute_energy_envelope(target_audio, fs, window)
    
    # Avoid division by zero
    target_envelope = np.maximum(target_envelope, 1e-10)
    
    # Compute gain curve
    gain = ref_envelope / target_envelope
    
    # Smooth the gain curve to avoid rapid changes
    if smoothing > 0 and len(gain) > 1:
        # Simple exponential moving average
        smoothed = np.zeros_like(gain)
        smoothed[0] = gain[0]
        alpha = 1 - smoothing
        for i in range(1, len(gain)):
            smoothed[i] = alpha * gain[i] + smoothing * smoothed[i-1]
        gain = smoothed
    
    # Limit gain range to avoid extreme changes
    gain = np.clip(gain, 0.5, 2.0)
    
    # Interpolate gain to sample level
    window_samples = int(window * fs)
    n_samples = len(target_audio)
    
    # Create time points for interpolation
    envelope_times = np.arange(len(gain)) * window_samples + window_samples // 2
    sample_times = np.arange(n_samples)
    
    # Interpolate
    if len(gain) > 1:
        interp_func = interp1d(envelope_times, gain, kind='linear', 
                               bounds_error=False, fill_value=gain[-1])
        gain_curve = interp_func(sample_times)
    else:
        gain_curve = np.ones(n_samples) * gain[0]
    
    # Apply gain curve to all channels
    for ch in range(target_audio.shape[1]):
        target_audio[:, ch] *= gain_curve
    
    return target_audio


def analyze_temporal_continuity(audio, fs, window=0.5):
    """Analyze how smooth or choppy the audio is.
    
    Parameters:
    - audio: Audio array (samples x channels)
    - fs: Sample rate
    - window: Analysis window in seconds
    
    Returns:
    - continuity_score: 0 (choppy) to 1 (smooth)
    - transition_type: "abrupt" or "gradual"
    - dynamic_range_db: Difference between quietest and loudest moments
    """
    # Compute energy envelope
    envelope = compute_energy_envelope(audio, fs, window)
    
    if len(envelope) < 2:
        return 1.0, "gradual", 0.0
    
    # Measure how quickly energy changes
    energy_diff = np.abs(np.diff(envelope))
    avg_change = np.mean(energy_diff)
    
    # High change = choppy, low change = smooth
    # Normalize to 0-1 range
    continuity_score = 1.0 / (1.0 + avg_change * 10)
    
    # Detect abrupt transitions (sudden energy changes)
    if len(energy_diff) > 0:
        threshold = np.percentile(energy_diff, 90)
        abrupt_count = np.sum(energy_diff > threshold)
        abrupt_ratio = abrupt_count / len(energy_diff)
        transition_type = "abrupt" if abrupt_ratio > 0.2 else "gradual"
    else:
        transition_type = "gradual"
    
    # Dynamic range
    energy_db = 20 * np.log10(envelope + 1e-15)
    dynamic_range_db = np.max(energy_db) - np.min(energy_db)
    
    return continuity_score, transition_type, dynamic_range_db
