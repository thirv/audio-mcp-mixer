# Spatial Audio Mixing Guide

## Cartesian Coordinate System

In spatial audio mixing, sound positions are represented using Cartesian coordinates `[x, y, z]`:

### Coordinate Definitions

- **x** (left-right): Horizontal left-right position
  - `-1` = fully left
  - `0` = center
  - `1` = fully right

- **y** (front-back): Horizontal front-back position
  - `1` = fully front
  - `0` = middle
  - `-1` = fully back

- **z** (up-down): Height position (elevation)
  - `0` = median plane (at ear level)
  - `1` = fully overhead

**Important:** All coordinate values must be numbers between -1 and 1.

## Common Position Examples

### Basic Positions
- `[ -1, 0, 0]` - Left
- `[  1, 0, 0]` - Right
- `[  0, 1, 0]` - Center front
- `[  0, -1, 0]` - Center back
- `[ -1, 1, 0]` - Left front
- `[  1, 1, 0]` - Right front
- `[ -1, -1, 0]` - Left back
- `[  1, -1, 0]` - Right back

### Overhead Positions
- `[ 0, 1, 1]` - Center front overhead
- `[ 0, -1, 1]` - Center back overhead
- `[-1,  0, 1]` - Left overhead
- `[ 1,  0, 1]` - Right overhead

### Intermediate Positions
You can use any values between -1 and 1 for precise positioning:
- `[-0.5, 0, 0]` - Slightly left of center
- `[0, 0.7, 0.3]` - Front center, slightly elevated
- `[0.3, -0.5, 0]` - Right and slightly back

## Speaker Layouts

### Stereo — `0+2+0` (2 channels)
> BS.2051 stereo speakers sit at ±30°, not the sides.
- Left: azimuth +30° → `[-0.5, 0.866, 0]`
- Right: azimuth -30° → `[0.5, 0.866, 0]`

### 5.1 Surround — `0+5+0` (6 channels)
- Front Left: azimuth +30° → `[-0.5, 0.866, 0]`
- Front Center: azimuth 0° → `[0, 1, 0]`
- Front Right: azimuth -30° → `[0.5, 0.866, 0]`
- Surround Left: azimuth +110° → `[-0.94, -0.342, 0]`
- Surround Right: azimuth -110° → `[0.94, -0.342, 0]`
- LFE (Subwoofer): Mono channel for low-frequency content (present in the layout, but left silent by this renderer)

### 7.1.4 Immersive — `4+7+0` (12 channels)
**Bed Layer (7 channels, elevation 0°):**
- Front Left: azimuth +30° → `[-0.5, 0.866, 0]`
- Front Center: azimuth 0° → `[0, 1, 0]`
- Front Right: azimuth -30° → `[0.5, 0.866, 0]`
- Side Left: azimuth +90° → `[-1, 0, 0]`
- Side Right: azimuth -90° → `[1, 0, 0]`
- Rear Left: azimuth +135° → `[-0.707, -0.707, 0]`
- Rear Right: azimuth -135° → `[0.707, -0.707, 0]`

**Height Layer (4 channels, elevation +30°):**
- Front Left Height: azimuth +45° → `[-0.612, 0.612, 0.5]`
- Front Right Height: azimuth -45° → `[0.612, 0.612, 0.5]`
- Rear Left Height: azimuth +135° → `[-0.612, -0.612, 0.5]`
- Rear Right Height: azimuth -135° → `[0.612, -0.612, 0.5]`

**LFE (Subwoofer):** Mono channel for low-frequency content (present in the layout, but left silent by this renderer)

## Mixing Philosophy: Front-Weighted by Default

Professional immersive audio mixes typically follow a front-weighted approach:

- **Front L/C/R channels (y≈1)** carry the primary content, main dialogue, key sound effects, and the central narrative. These are your "hero" speakers.
- **Surround channels (sides and rear)** add width, depth, and ambience. They create the sense of space.
- **Height channels (overhead)** add dimension and immersion — rain, thunder, aircraft, atmospheric effects.

**When to break the rule:** The scene dictates placement. A plane flying overhead should be in the height channels. A monster sneaking up from behind should be in the rear. The front-weighted approach is your default tendency, not a restriction.

## Mixing Guidelines by Sound Type

### Music
- Typically mixed toward the front unless specified otherwise
- Bass and drums: Center to slightly left/right
- Vocals: Center front
- Guitars: Left and right front
- Keyboards: Wide stereo spread
- Effects: Can be panned to sides or overhead

### Environmental Sounds
- Typically mixed to the left and right sides or overhead
- Wind: Overhead and rear speakers
- Rain: Surround and overhead for immersive effect
- Thunder: Overhead with some front presence

### Sound Effects
- Transient sounds (explosions, impacts): Front center or overhead
- Whooshes/movement: Can travel from front to back
- Continuous mechanical: Side speakers
- Footsteps: Can indicate movement through panning

### Voices/Dialogue
- Primary dialogue: Center front
- Secondary voices: Slightly left/right front
- Crowd ambience: Surround speakers
- Off-screen voices: Surround speakers

## Practical Positioning Tips

### Creating Depth
Use front/back position (y) as a directional anchor:
1. **Foreground:** Front speakers (y positive, up to +1 for front center)
2. **Midground:** Side speakers (y ≈ 0)
3. **Background:** Rear speakers (y negative, roughly −0.34 to −0.71)

### Creating Height
1. **Ground level:** z = 0
2. **Elevated:** z = 0.3-0.7
3. **Overhead:** z = 1.0

### Panning Movement
When a sound should move, use a trajectory with multiple keyframes:
- Use `[(time_1, x_1, y_1, z_1), (time_2, x_2, y_2, z_2)]` for smooth movement
- For staggered start times, use `[time, x, y, z]` (4 numbers) to delay a sound
- For simultaneous sounds at different positions, use `[x, y, z]` (3 numbers)

### Avoiding Spectral Masking
When multiple sounds occupy similar frequency ranges:
1. Spatial separation: Place them at different positions
2. Height layering: Put one at ground level, another overhead
3. Front/back separation: One front, one back

## Common Mistakes to Avoid

1. **All sounds at center front:** This sounds unnatural and flat. Use the full 3D space.
2. **Ignoring height:** For immersive formats, use overhead speakers to create depth.
3. **Too much reverb:** Overhead speakers are for height, not just for reverb.
4. **Inconsistent panning:** Keep movement smooth and logical.
5. **Relying on the LFE for bass:** This direct renderer leaves the LFE (subwoofer) channel silent by design, so low-frequency content is not routed to the subwoofer — expect your playback system's bass management to handle it.

## Advanced Techniques

### Creating 3D Soundscapes
1. Layer sounds at different heights
2. Use frequency analysis to ensure full spectral coverage
3. Consider how sounds naturally occur in real environments
4. Use quiet segments for subtle details

### Mix Analysis and Mimicry
Use `analyze_mix_spatial` to understand existing mixes:
- `analyze_mix_spatial(mix_file)` - Get analysis with sounds, positions, levels (level_db), and spatial balance

Use `mimic_mix_direct` to create similar mixes:
- `mimic_mix_direct(mix_file)` - Create a new mix with similar spatial distribution

### Dynamic Mixing
Change positions over time by:
- Using multiple sound instances with different offsets
- Creating variations of sounds at different positions
- Layering ambient elements at different spatial locations
