"""
EAR-based direct audio renderer for spatial audio mixing.

This module provides ``EarDirectRenderer``, a thin object-based renderer built on the
EBU ADM Renderer (EAR). It lets callers place mono sounds at static positions or along
time-stamped trajectories and render them to a standard loudspeaker layout without
requiring an ADM file.
"""

import math

import numpy as np

try:
    from ear.core import bs2051, point_source
    _EAR_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional runtime
    bs2051 = None
    point_source = None
    _EAR_AVAILABLE = False

# Public tool vocabulary -> EAR (BS.2051) layout name.
LAYOUT_ALIASES = {
    'stereo': '0+2+0',
    'surround': '0+5+0',
    'immersive': '4+7+0',
    '2.0.0': '0+2+0',
    '5.1.0': '0+5+0',
    '7.1.4': '4+7+0',
}

def resolve_layout_name(layout_name: str) -> str:
    """Map a friendly layout name to the EAR/BS.2051 layout name."""
    key = str(layout_name).strip().lower()
    return LAYOUT_ALIASES.get(key, layout_name)

def cart2polar(x: float, y: float, z: float):
    """
    Convert ADM-format Cartesian coordinates to polar (azimuth, elevation, distance).

    ADM convention (see ear.common.azimuth/elevation): x=+1 (right) -> -90 deg,
    y=+1 (front) -> 0 deg, x=-1 (left) -> +90 deg. Distance is always 1.0.
    """
    x, y, z = float(x), float(y), float(z)
    if x == 0.0 and y == 0.0 and z == 0.0:
        return (0.0, 0.0, 1.0)
    azimuth = -math.degrees(math.atan2(x, y))
    elevation = math.degrees(math.atan2(z, math.hypot(x, y)))
    return (azimuth, elevation, 1.0)

class EarDirectRenderer:
    """Direct object-based audio renderer using the EBU ADM Renderer (EAR)."""

    def __init__(self, layout_name: str, sample_rate: int = 48000, lfe_gain_db: float = None,
                 use_hf: bool = False):
        """Create a renderer for ``layout_name``.

        Args:
            layout_name: Public mix type ('stereo'/'surround'/'immersive') or an EAR name.
            sample_rate: Output sample rate in Hz.
            lfe_gain_db: Optional LFE gain in dB (unused while LFE is silent by default).
            use_hf: **Compatibility no-op.** EAR 2.1.0 exposes a single point-source panner
                (``ear.core.point_source``, a frequency-independent VBAP panner) and has no
                ``point_source_hf``. This flag is accepted so code written against the older
                spec does not fail, but it does not change behaviour: elevation-aware panning
                is already handled by the standard panner.
        """
        if not _EAR_AVAILABLE:
            raise RuntimeError(
                "The 'ear' package is required for EarDirectRenderer but could not be imported."
            )
        resolved = resolve_layout_name(layout_name)
        try:
            layout = bs2051.get_layout(resolved)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Unknown layout name {layout_name!r} (resolved to {resolved!r}). "
                "Use 'stereo', 'surround', 'immersive' or an EAR layout name such as '0+5+0'."
            ) from exc
        self.layout_name = layout_name
        self.ear_layout_name = resolved
        self.sample_rate = sample_rate
        self.lfe_gain_db = lfe_gain_db
        self.use_hf = use_hf  # compat no-op; EAR 2.1.0 has a single point-source panner
        self.layout = layout
        self.n_channels = len(layout.channels)
        self.lfe_index = int(np.argmax(layout.is_lfe)) if bool(np.any(layout.is_lfe)) else None
        # Use without_lfe when available; fall back to the full layout
        layout_for_panner = getattr(layout, 'without_lfe', layout)
        self._panner = point_source.configure(layout_for_panner)
        self._objects = []

    @property
    def audio_objects(self):
        """Backwards-compatible alias for the queued object list."""
        return self._objects

    def add_audio_object(self, path: str, positions: list, gain: float = 1.0,
                         track_index: int = None, name: str = "") -> int:
        positions = self._normalize_positions(positions)
        self._objects.append({
            'path': path, 'positions': positions, 'gain': gain,
            'track_index': track_index, 'name': name,
        })
        return len(self._objects) - 1

    @staticmethod
    def _normalize_positions(positions) -> list:
        if not isinstance(positions, (list, tuple)) or len(positions) == 0:
            raise ValueError("positions must be a non-empty list or tuple")
        first = positions[0]
        if isinstance(first, (list, tuple)):
            if len(positions) == 1:
                # A single position wrapped in a list: [(x, y, z)] or [(t, x, y, z)].
                single = tuple(float(v) for v in positions[0])
                keyframes = [single] if len(single) == 4 else [(0.0,) + single]
            else:
                # A list of keyframes: [(t, x, y, z), ...].
                keyframes = [tuple(float(v) for v in kf) for kf in positions]
        elif len(positions) == 4:
            keyframes = [tuple(float(v) for v in positions)]
        elif len(positions) == 3:
            keyframes = [(0.0,) + tuple(float(v) for v in positions)]
        else:
            raise ValueError(
                "positions must be (x, y, z), (time, x, y, z), or a list of keyframes"
            )
        for kf in keyframes:
            if len(kf) != 4:
                raise ValueError(f"each keyframe must be (time, x, y, z); got {kf!r}")
        return keyframes

    def render(self, output_path: str, normalize: bool = True) -> str:
        """
        Render the queued audio objects to ``output_path``.

        Uses the EAR point-source panner to spatially mix the audio objects according to
        their positions and the specified layout.  Gains are interpolated over time in the
        panner's native Cartesian domain, then expanded to the full channel count
        (re-inserting a silent LFE column when the layout has one).
        """
        import soundfile as sf
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)

        # ------------------------------------------------------------------
        # Shared input loader: each file is read at most once per render (the
        # duration probe below and the render pass both call this) and is
        # resampled to the renderer's sample rate when the input differs.
        # ------------------------------------------------------------------
        read_cache = {}

        def _load(path):
            cached = read_cache.get(path)
            if cached is not None:
                return cached
            data, sr = sf.read(path, dtype='float32')
            if sr != self.sample_rate:
                from scipy.signal import resample_poly
                src_sr = sr
                data = resample_poly(data, self.sample_rate, sr, axis=0)
                sr = self.sample_rate
                logger.info(
                    "Resampled %s from %d Hz to %d Hz", path, src_sr, self.sample_rate
                )
            read_cache[path] = (data, sr)
            return data, sr

        # ------------------------------------------------------------------
        # Phase 1: determine output duration from all queued objects
        # ------------------------------------------------------------------
        max_end = 0.0
        for obj in self._objects:
            data, sr = _load(obj['path'])
            dur = len(data) / sr
            last_time = obj['positions'][-1][0]
            end = last_time + dur
            if end > max_end:
                max_end = end

        duration = max(max_end, 0.001)  # avoid zero-length
        n_samples = int(duration * self.sample_rate)
        n_ch_no_lfe = self.n_channels - 1 if self.lfe_index is not None else self.n_channels

        # ------------------------------------------------------------------
        # Phase 2: render each object into the full output buffer
        # ------------------------------------------------------------------
        output = np.zeros((n_samples, self.n_channels), dtype=np.float32)

        for obj in self._objects:
            data, sr = _load(obj['path'])
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            keyframes = obj['positions']  # list of (t, x, y, z)
            start_time = keyframes[0][0]
            n_audio = len(data)
            sound_dur = n_audio / sr

            # zero-pad buffer for this object (covers delayed start)
            obj_buf = np.zeros((n_samples, n_ch_no_lfe), dtype=np.float32)

            # Build time-aligned keyframe list covering [start_time, start_time + sound_dur]
            kf_times = [kf[0] for kf in keyframes]
            kf_pos = [(kf[1], kf[2], kf[3]) for kf in keyframes]

            t_begin = max(0.0, start_time)
            t_end = start_time + sound_dur

            if len(keyframes) == 1:
                # Static position: compute gains once
                x, y, z = kf_pos[0]
                pos = np.array([x, y, z], dtype=np.float64)
                gains = self._panner.handle(pos)
                if gains is None:
                    logger.warning(
                        "panner.handle returned None for position %s -> zeroing gains", pos
                    )
                    gains = np.zeros(n_ch_no_lfe, dtype=np.float64)
            else:
                # Moving: linearly interpolate position over the sound duration,
                # evaluate the panner at densified steps, then linearly
                # interpolate the gains up to sample-rate resolution (below).
                densify = 100
                total_steps = max(1, int(sound_dur * self.sample_rate / densify))
                step_times = np.linspace(t_begin, t_end, total_steps)
                gains_array = np.zeros((total_steps, n_ch_no_lfe), dtype=np.float64)
                for i, t in enumerate(step_times):
                    # Find surrounding keyframes and interpolate position
                    if t <= kf_times[0]:
                        px, py, pz = kf_pos[0]
                    elif t >= kf_times[-1]:
                        px, py, pz = kf_pos[-1]
                    else:
                        for k in range(len(kf_times) - 1):
                            if kf_times[k] <= t <= kf_times[k + 1]:
                                t1, t2 = kf_times[k], kf_times[k + 1]
                                if t2 != t1:
                                    ratio = (t - t1) / (t2 - t1)
                                else:
                                    ratio = 0.0
                                px = kf_pos[k][0] + ratio * (kf_pos[k + 1][0] - kf_pos[k][0])
                                py = kf_pos[k][1] + ratio * (kf_pos[k + 1][1] - kf_pos[k][1])
                                pz = kf_pos[k][2] + ratio * (kf_pos[k + 1][2] - kf_pos[k][2])
                                break
                        else:
                            px, py, pz = kf_pos[0]
                    pos = np.array([px, py, pz], dtype=np.float64)
                    g = self._panner.handle(pos)
                    if g is None:
                        g = np.zeros(n_ch_no_lfe, dtype=np.float64)
                    gains_array[i] = g

                # Upsample gains to sample-rate resolution via LINEAR interpolation
                # (T1.2: "densify + linearly interpolate gains over time"). This is
                # what makes moving-object trajectories smooth instead of zipping.
                sample_steps = min(n_audio, n_samples)
                if sample_steps == 0:
                    continue
                src_idx = np.arange(total_steps)
                dst_idx = np.linspace(0.0, total_steps - 1, sample_steps)
                sample_gains = np.empty((sample_steps, n_ch_no_lfe), dtype=np.float64)
                for c in range(n_ch_no_lfe):
                    sample_gains[:, c] = np.interp(dst_idx, src_idx, gains_array[:, c])

                # Apply per-sample gains, offset to start_time
                offset = max(0, int(start_time * self.sample_rate))
                length = min(n_audio, n_samples - offset)
                if length > 0:
                    audio_part = data[:length]
                    gain_part = sample_gains[:length]
                    obj_buf[offset:offset + length] += gain_part * audio_part
                # re-insert LFE column and accumulate into output
                if self.lfe_index is not None:
                    tmp = np.zeros((n_samples, self.n_channels), dtype=np.float32)
                    tmp[:, :self.lfe_index] = obj_buf[:, :self.lfe_index]
                    tmp[:, self.lfe_index + 1:] = obj_buf[:, self.lfe_index:]
                    output += tmp * obj['gain']
                else:
                    output += obj_buf * obj['gain']
                continue  # skip the static block below

            # Static path: apply uniform gains to whole audio
            gains = gains * obj['gain']
            offset = max(0, int(start_time * self.sample_rate))
            length = min(n_audio, n_samples - offset)
            if length > 0:
                audio_part = data[:length]
                obj_buf[offset:offset + length] += gains * audio_part

            # re-insert LFE column and accumulate into output
            if self.lfe_index is not None:
                tmp = np.zeros((n_samples, self.n_channels), dtype=np.float32)
                tmp[:, :self.lfe_index] = obj_buf[:, :self.lfe_index]
                tmp[:, self.lfe_index + 1:] = obj_buf[:, self.lfe_index:]
                output += tmp
            else:
                output += obj_buf

        # ------------------------------------------------------------------
        # Phase 4: normalise peak to 0.9
        # ------------------------------------------------------------------
        if normalize:
            peak = np.max(np.abs(output))
            if peak > 0:
                output = output * 0.9 / peak

        # ------------------------------------------------------------------
        # Phase 5: write
        # ------------------------------------------------------------------
        sf.write(output_path, output, self.sample_rate)
        return output_path


__all__ = ['EarDirectRenderer', 'cart2polar', 'resolve_layout_name', 'LAYOUT_ALIASES']
