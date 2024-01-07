Drum hit timestamp finder

# Strategy

infers whether the hit or not at the middle of window.
input: pcm window, hits on a left side of half of window.
output: hit or not at the middle of window

# Generate train dataset

## Audio samples

Downloads free drum instrument sample audio files. Thank you internet!

## Preprocessing audio samples

Audio samples would need trim their front and back silent parts.

Trims by 'pcm value > threshold' and give constant little padding on front.

## Generate random dataset

generates audio file by randomly select below criteria.

- bpm
  - constant bpm
  - dynamic bpm
    - 1~2 changes
- beat length
- kick: 0~2
- cymbals: 0~4
- snares: 0~3
- beat type
  - presets
    - manually made
  - fully random
    - put enough padding between hits for realistic

dataset should contains the timings of hit together for training.

# Model architecture

- Window size: x samples ('sample' in audio pcm file, like 'sample rate')
- Sample type: f32
- Input: x samples and front-half hit information
