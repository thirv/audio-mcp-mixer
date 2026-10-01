"""
Tests for EarDirectRenderer and cart2polar.

Covers the T1.3 requirements:
  * cart2polar ADM convention
  * layout alias resolution + channel counts (2 / 6 / 12)
  * LFE column stays silent by default
  * verified immersive gain map (PLAN.md reference)
  * moving objects sweep energy across speakers (linear gain interpolation, T1.2)
  * delayed start produces leading silence
  * peak normalization

All audio output is written under pytest's tmp_path -- nothing is written to CWD.
"""

import numpy as np
import pytest
import soundfile as sf

from ear_direct_renderer import EarDirectRenderer, cart2polar


# ---------------------------------------------------------------------------
# cart2polar (ADM convention)
# ---------------------------------------------------------------------------

def test_cart2polar_adm_convention():
    az, el, d = cart2polar(1.0, 0.0, 0.0)
    assert az == pytest.approx(-90.0)
    assert el == pytest.approx(0.0)
    assert d == pytest.approx(1.0)


def test_cart2polar_zero_vector():
    assert cart2polar(0.0, 0.0, 0.0) == (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Layout resolution
# ---------------------------------------------------------------------------

def test_layout_alias_resolution():
    for alias, expected in [
        ("stereo", 2), ("0+2+0", 2),
        ("surround", 6), ("5.1.0", 6), ("0+5+0", 6),
        ("immersive", 12), ("7.1.4", 12), ("4+7+0", 12),
    ]:
        assert EarDirectRenderer(alias).n_channels == expected


def test_invalid_layout_raises():
    with pytest.raises(ValueError):
        EarDirectRenderer("bogus-layout")


def test_use_hf_kwarg():
    assert EarDirectRenderer("immersive", use_hf=False).n_channels == 12


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_static(layout, wav, position, out_path, gain=1.0, normalize=True):
    """Add a single static audio object and render to out_path; return the renderer."""
    r = EarDirectRenderer(layout)
    r.add_audio_object(wav, [position], gain)
    r.render(str(out_path), normalize=normalize)
    return r


# ---------------------------------------------------------------------------
# Render correctness
# ---------------------------------------------------------------------------

def test_static_render_non_silent(tmp_path, mono_test_wav):
    out = tmp_path / "static.wav"
    _render_static("surround", mono_test_wav, (0.0, 1.0, 0.0, 0.0), out)
    data, sr = sf.read(str(out), dtype="float32")
    assert sr == 48000
    assert data.shape[1] == 6
    assert np.abs(data).max() > 0


def test_render_channel_counts(tmp_path, mono_test_wav):
    for layout, expected in [("stereo", 2), ("surround", 6), ("immersive", 12)]:
        out = tmp_path / f"ch_{layout}.wav"
        _render_static(layout, mono_test_wav, (0.0, 0.5, 0.0, 0.0), out)
        assert sf.info(str(out)).channels == expected


def test_render_lfe_silent(tmp_path, mono_test_wav):
    for layout in ("surround", "immersive"):
        out = tmp_path / f"lfe_{layout}.wav"
        r = _render_static(layout, mono_test_wav, (0.0, 0.5, 0.0, 0.0), out)
        assert r.lfe_index is not None
        data, _ = sf.read(str(out), dtype="float32")
        # LFE column is driven only by an explicit lfe channel, so it stays silent.
        assert np.abs(data[:, r.lfe_index]).max() == 0.0
        # Non-LFE channels carry the signal.
        non_lfe = np.delete(data, r.lfe_index, axis=1)
        assert np.abs(non_lfe).max() > 0


def test_render_gain_map_immersive(tmp_path, mono_test_wav):
    # PLAN.md verified-gain reference for the 4+7+0 immersive layout.
    cases = [
        ((1.0, 0.0, 0.0), 5),   # M-090
        ((-1.0, 0.0, 0.0), 4),  # M+090
        ((0.0, 1.0, 0.0), 2),   # M+000
    ]
    for xyz, expected_ch in cases:
        out = tmp_path / f"gm_{xyz[0]}_{xyz[1]}_{xyz[2]}.wav"
        _render_static("immersive", mono_test_wav, (0.0, *xyz), out)
        data, _ = sf.read(str(out), dtype="float32")
        energy = (data ** 2).sum(axis=0)
        assert int(np.argmax(energy)) == expected_ch


def test_render_moving_object_sweeps(tmp_path, mono_test_wav_2s):
    # A 2 s tone sweeping left (-x) -> right (+x) in immersive (4+7+0).
    out = tmp_path / "sweep.wav"
    r = EarDirectRenderer("immersive")
    r.add_audio_object(
        mono_test_wav_2s,
        [(0.0, -1.0, 0.0, 0.0), (2.0, 1.0, 0.0, 0.0)],
        gain=1.0,
    )
    r.render(str(out), normalize=True)
    data, sr = sf.read(str(out), dtype="float32")
    assert data.shape[1] == 12
    energy = (data ** 2).sum(axis=0)
    assert energy[4] > 0 and energy[5] > 0  # M+090 and M-090 both lit over the sweep

    # The tone plays during [0, sound_dur] (the keyframe span), while the
    # renderer pads the buffer out to last_keyframe_time + sound_dur. So
    # sample the "early" / "late" windows within the actual audio span --
    # the trailing region of the buffer is silent padding, not the sweep tail.
    n_audio = int(sf.info(str(mono_test_wav_2s)).duration * sr)
    seg = sr // 4  # 0.25 s
    early = data[:seg]
    late = data[n_audio - seg : n_audio]
    # Left speaker dominates early, right speaker dominates late (energy moved).
    assert np.sum(early[:, 4] ** 2) > np.sum(late[:, 4] ** 2)
    assert np.sum(late[:, 5] ** 2) > np.sum(early[:, 5] ** 2)


def test_render_delayed_start_silence(tmp_path, mono_test_wav_2s):
    out = tmp_path / "delayed.wav"
    _render_static("surround", mono_test_wav_2s, (1.0, 0.5, 0.0, 0.0), out)
    data, sr = sf.read(str(out), dtype="float32")
    assert sr == 48000
    assert np.abs(data[:sr]).max() == 0.0      # 1 s of leading silence
    assert np.abs(data[sr:]).max() > 0          # signal after the delay


def test_render_normalize_peak(tmp_path, mono_test_wav):
    out = tmp_path / "norm.wav"
    _render_static("immersive", mono_test_wav, (0.0, 0.0, 1.0, 0.0), out, normalize=True)
    data, _ = sf.read(str(out), dtype="float32")
    peak = np.abs(data).max()
    assert peak > 0
    # The 0.9 target is quantized to PCM_16 on disk, so allow ~1 LSB of slack.
    assert 0.85 <= peak <= 0.9 + 1e-3


def test_render_no_normalize(tmp_path, mono_test_wav):
    out = tmp_path / "nonorm.wav"
    _render_static("surround", mono_test_wav, (0.0, 0.5, 0.0, 0.0), out, normalize=False)
    data, _ = sf.read(str(out), dtype="float32")
    assert data.shape[1] == 6
    assert np.abs(data).max() > 0


# ---------------------------------------------------------------------------
# C.1a: sample-rate guard + single shared read (Phase 1 and Phase 2 share it)
# ---------------------------------------------------------------------------

def test_render_resamples_to_sample_rate(tmp_path):
    # Feed a 44.1 kHz input and assert the renderer resamples it to 48 kHz.
    sr_in, sr_out = 44100, 48000
    duration = 0.5
    n = int(duration * sr_in)
    t = np.arange(n) / sr_in
    wave = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = tmp_path / "in_441.wav"
    sf.write(str(wav), wave, sr_in, subtype="PCM_16")

    out = tmp_path / "out_48k.wav"
    _render_static("surround", str(wav), (0.0, 0.5, 0.0, 0.0), out)

    data, osr = sf.read(str(out), dtype="float32")
    assert osr == sr_out
    assert np.abs(data).max() > 0
    # Duration is preserved through resampling (within a couple of samples).
    assert abs(len(data) - duration * sr_out) <= 2


def test_render_reads_each_file_once(tmp_path, mono_test_wav, monkeypatch):
    # Phase 1 (duration probe) and Phase 2 (render) must share one read of each
    # input file. Count calls to soundfile.read and assert a single read.
    real_read = sf.read
    counts = {}

    def counting_read(*args, **kwargs):
        key = str(args[0]) if args else str(kwargs.get("file"))
        counts[key] = counts.get(key, 0) + 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(sf, "read", counting_read)

    out = tmp_path / "once.wav"
    _render_static("surround", mono_test_wav, (0.0, 0.5, 0.0, 0.0), out)

    assert counts.get(str(mono_test_wav)) == 1