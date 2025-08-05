import RPi.GPIO as GPIO
import time
import subprocess
import torchaudio
import torchaudio.functional as F
import torch
from asteroid.models import DPRNNTasNet
import warnings
import os
import signal

warnings.filterwarnings("ignore")

# GPIO Pins
BP1 = 16  # Record
BP2 = 20  # Play Source_1.wav
BP3 = 21  # Play Source_2.wav

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(BP1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BP2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BP3, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Load model
print("Loading model...")
model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean", map_location='cpu')
model.eval()
print("Model loaded.")

recording_proc = None
recording_active = False

def separate_sources(filename="MixedSpeech.wav"):
    print("Separating sources...")
    waveform, sr = torchaudio.load(filename)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 8000:
        waveform = F.resample(waveform, sr, 8000)
        sr = 8000
    waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        est_sources = model.separate(waveform)
        est_sources = est_sources.squeeze(0)

    for i in range(est_sources.shape[0]):
        torchaudio.save(f"Source_{i+1}.wav", est_sources[i].unsqueeze(0), sr)
        print(f"Saved: Source_{i+1}.wav")
    print("Separation complete.")

def play_audio(file):
    print(f"Playing {file}")
    subprocess.run(["aplay", "-D", "pulse", file])

try:
    while True:
        bp1_state = GPIO.input(BP1)
        bp2_state = GPIO.input(BP2)
        bp3_state = GPIO.input(BP3)

        # Record on press-and-hold
        if bp1_state == GPIO.LOW and not recording_active:
            print("Recording started...")
            recording_proc = subprocess.Popen([
                "arecord", "-D", "pulse",
                "-f", "S16_LE",  # 16-bit
                "-t", "wav",
                "-r", "16000",   # 16 kHz
                "-c", "1",       # Mono
                "MixedSpeech.wav"
            ])
            recording_active = True

        elif bp1_state == GPIO.HIGH and recording_active:
            recording_proc.terminate()
            recording_proc.wait()
            print("Recording stopped.")
            recording_active = False
            separate_sources()

        # Playbacks
        if bp2_state == GPIO.LOW:
            play_audio("Source_1.wav")

        if bp3_state == GPIO.LOW:
            play_audio("Source_2.wav")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
    if recording_proc:
        recording_proc.terminate()
    GPIO.cleanup()