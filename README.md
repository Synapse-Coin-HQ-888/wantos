# Watmos

*Audio-Reactive Sampling for LLMs*

This repository demonstrates a proof-of-concept technique for steering Large Language Model (LLM) decoding by modulating the sampling temperature in real time using the loudness profile of a music track.

## Core Idea

We hypothesize that human music embodies the brain’s native stochastic drive. By coupling this variability to the temperature parameter, the model explores its latent space under a human-scaled rhythm.

Practically, the loudness trace of a song functions like a pre-sampled probability signal—an external proxy for a brain-like softmax over universal pattern prediction. We align the LLM’s sampling to that rendered temporal distribution.

In its current form, the project lets you compare how different musical styles—jazz, techno, or even silence as a control—shape text generation when temperature is driven by the track.

## Pipeline Overview

The script transforms an audio file into a loudness curve and maps it onto temperature for token-wise generation.

1. **Audio Analysis** — The input track is loaded, and LUFS loudness is measured per frame at a rate that matches tokens-per-second (TPS).
2. **Velocity Estimation** — The loudness series is smoothed, then differentiated to obtain a “loudness velocity,” capturing dynamic intensity.
3. **Temperature Mapping** — Velocity is linearly (or nonlinearly) mapped to a temperature range: calmer passages yield lower, safer temperatures; energetic peaks drive higher, more exploratory temperatures.
4. **Reactive Inference** — A local LLM server (OpenAI-compatible) receives the rolling history and the current temperature for each emitted token.
5. **Interactive Playback** — A terminal TUI renders text in sync with music playback. You can pause, seek, and restart on the fly.
6. **Run Persistence** — Prompts, analysis outputs, and token streams can be saved to disk and replayed later.
