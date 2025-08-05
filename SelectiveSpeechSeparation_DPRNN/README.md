# Selective Speech Separation on Raspberry Pi using DPRNN TasNet

- This project implements a **real-time selective speech separation system** on a **Raspberry Pi 4B** using the **DPRNN-TasNet** model from the [Asteroid toolkit](https://github.com/asteroid-team/asteroid). 
- It demonstrates the power of **Edge AI** by deploying a pre-trained deep learning model on a low-power, affordable embedded system to separate mixed audio signals into individual speech sources.

---

## Overview

Selective Speech Separation is the task of isolating individual speakers from a single-channel audio recording containing overlapping speech. This project enables real-time recording, separation, and playback of isolated sources with the press of physical buttons on a Raspberry Pi.

---

## Features

-  Record mixed speech using a microphone
-  Separate the audio into individual speakers using DPRNN-TasNet
-  Playback separated sources through physical push buttons
-  Runs entirely on Raspberry Pi (Edge Computing)
-  Uses `torchaudio`, `asteroid`, and `PyTorch` for audio and model processing

---

## Hardware Requirements

- Raspberry Pi 4B (4GB recommended)
- USB microphone or supported audio input device
- 3x Push Buttons (GPIO-connected)
- Speaker or 3.5mm audio output device

---

## Software Stack

- Python 3.7+
- PyTorch
- torchaudio
- asteroid
- RPi.GPIO
- arecord / aplay (ALSA)
- RealVNC (optional for GUI access)
- PuTTY (optional for remote CLI access)

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Selective-Speech-Separation-DPRNN.git
   cd Selective-Speech-Separation-DPRNN
   ```

## Install Dependencies
```bash
pip install torch torchaudio asteroid RPi.GPIO
```

## Enable Audio & GPIO Support
- Make sure ALSA is configured and working on Raspberry Pi.
- Connect the push buttons to GPIO pins 16, 20, and 21 as input with pull-up resistors.

## How it works
- BP1 (GPIO 16) – Press and hold to start recording. Release to stop and trigger separation.
- BP2 (GPIO 20) – Press to play the first separated source.
- BP3 (GPIO 21) – Press to play the second separated source.

  Recording is done using arecord and played using aplay. Upon stopping the recording, the audio is separated using the pre-trained DPRNN-TasNet model (2-speaker, WHAM! dataset).

## File Structure
```bash
├── main.py               # Main application script
├── Source_1.wav          # Output audio source 1
├── Source_2.wav          # Output audio source 2
├── MixedSpeech.wav       # Recorded mixed input
└── README.md             # Project documentation
```

## Future Improvements

- Add LED indicators for recording/playback states
- Support multi-speaker separation (>2)
- Optimize model inference for faster separation
- Integrate with a small display for UI feedback
