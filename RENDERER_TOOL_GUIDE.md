# Renderer Tool - Immersive Audio Mixing API

A direct rendering API for immersive audio mixing that bypasses ADM file parsing and provides programmatic control over object-based audio rendering using the EAR audio renderer. The high-level mixing interface is implemented in the tool `render_spatial_mix`

**Use `render_spatial_mix` for:**
- Static sounds in desired positions
- Static routing of sounds to specific output channels
- Moving sounds (vehicles, flying objects, dynamic elements)
- When specific positions or trajectories are requested
- Experimental spatial audio with dynamic object movement
- Creating dense mixes by combining the above in a single call, there is no limit to number of sounds in the mix

### Understanding Traditional vs. Spatial Mixing

**Traditional Immersive Mixing:**
- Most sound energy and content is in the stereo L/R channels (frontal stereo image)
- Unless a position is specifically requested for a certain sound, it typically belongs to the frontal stereo image
- Objects and other channels are used to enhance the experience with spatial effects
- When a position or trajectory is requested, the user typically wants to utilize the full spatial range
- Sounds can be placed anywhere in 3D space or move along trajectories

## Quick Start

```python
from tools import render_spatial_mix

# Static sounds (simple [x, y, z] format)
render_spatial_mix(
    file_in=['drums', 'bass'],
    mix_type='immersive',
    position=[[-1, 0, 0], [1, 0, 0]]
)

# Moving sound - helicopter flying top left to top right
render_spatial_mix(
    file_in=['helicopter'],
    mix_type='immersive',
    position=[[(0, -1, 0, 1), (5, 1, 0, 1)]],
)
```

## API Reference

### render_spatial_mix()

**Parameters:**
- `file_in` (list): Audio file paths without .wav extension
- `mix_type` (str): Output format - 'stereo', 'surround', or 'immersive'
- `position` (list): List of positions or trajectories, one per input file
- `suffix` (str, optional): Unique suffix for output filename
- `fade_duration` (float, optional): Fade in/out duration in seconds (default: 0, no fade)

**Returns:** `(result_message, error_flag)` tuple

#### Cartesian Coordinates

- x: -1 (left) to 1 (right)
- y: 1 (front) to -1 (back)
- z: 0 (middle) to 1 (overhead)

#### Position Format

The `position` parameter accepts three formats, one per input file:

**1. Static sound at t=0:** `[x, y, z]` (3 numbers)

```python
position = [[-1, 0, 0], [1, 0, 0]]  # Two static sounds: left and right
```

**2. Static sound with delayed start:** `[time, x, y, z]` (4 numbers)

```python
position = [[5, -1, 0, 0]]  # Sound starts at 5 seconds, positioned left
```

**3. Moving trajectory:** `[(time_1, x_1, y_1, z_1), ...]` (list of keyframes)

```python
position = [[(0, -1, 0, 1), (5, 1, 0, 1)]]  # Moving left to right overhead
```

**Mixed example (static + delayed + moving):**

```python
position = [
    [-1, 0, 0],                          # Static at left
    [5, 1, 0, 0],                         # Delayed start at 5s, right
    [(0, -1, 0, 1), (5, 1, 0, 1)]         # Moving left to right overhead
]
```

## Example Workflows

### 1. Forest Scene (5.1 Surround)

```python
# Generate audio
generate_single_sound('birds', dur=3.0)
generate_single_sound('wind', dur=3.0)
generate_single_sound('stream', dur=3.0)

# Create surround mix
render_spatial_mix(
    file_in=['birds', 'wind', 'stream'],
    mix_type='surround',
    position=[[0, 1, 0], [0, -1, 0], [1, 0, 0]]
)
```

### 2. Chase Scene (Moving Objects)

```python
# Both cars moving in opposite directions
render_spatial_mix(
    file_in=['car1', 'car2'],
    mix_type='immersive',
    position=[
        [(0, -1, 0, 0), (2.5, 0, 0, 0), (5, 1, 0, 0)],   # Car 1: left to right
        [(0, 1, 0, 0), (2.5, 0, 0, 0), (5, -1, 0, 0)]    # Car 2: right to left
    ]
)
```

### 3. City Street Ambience (Mixed Static/Moving)

```python
# Static traffic and crowd, moving siren
render_spatial_mix(
    file_in=['traffic', 'siren', 'crowd'],
    mix_type='immersive',
    position=[
        [0, -1, 0],                                   # Static: traffic behind
        [(0, -1, 0, 1), (10, 1, 0, 1)],               # Moving: siren left to right
        [0, 1, 0]                                      # Static: crowd in front
    ]
)
```

### 4. Horror Scene with Staggered Timing

```python
# Multiple sounds at different start times in a single call
render_spatial_mix(
    file_in=['scream', 'gunshot', 'chainsaw', 'ambience'],
    mix_type='immersive',
    position=[
        [0, 0, 1, 0],           # Scream starts immediately, front center
        [5, -1, 0, 0],          # Gunshot at 5 seconds, left
        [10, 1, 0, 0],          # Chainsaw at 10 seconds, right
        [0, 0, -1, 0.3]         # Ambience throughout, behind and slightly elevated
    ],
    fade_duration=0.1        # Optional: smooth fade in/out
)
```

## Current Limitations

### LFE (Low Frequency Effects) Channel

**Current behavior:** The LFE channel is currently silent in all rendered outputs.

**Future support:** LFE output could be added as a post-processing step after the main mix is complete:
1. Low-pass filter each channel (typically below 80-120 Hz)
2. Route the filtered low-frequency content to the LFE channel

For now, bass management is handled by the playback system.

## Troubleshooting

### Silent Output

1. Verify audio files exist and have correct sample rate (48kHz)
2. Check position coordinates are valid (-1 to 1)
3. Ensure gains are positive values

### File Not Found Error

Generate the audio file first or check the filename spelling.

## Dependencies

- numpy
- soundfile
- ear (included in project)

## Credits

Built on top of the EAR audio renderer by EBU.
