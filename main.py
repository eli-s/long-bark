import os
import torch
import soundfile as sf
import nltk  # we'll use this to split into sentences
import numpy as np

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device for PyTorch operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

preload_models()


nltk.download('punkt')


# load the script from script.txt file
with open("script.txt", "r") as f:
    script = f.read()

script = script.replace("\n", " ").strip()

# split the script into sentences
sentences = nltk.sent_tokenize(script)

GEN_TEMP = 0.6
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
idx = 0
for sentence in sentences:
    print(f"Generating audio for sentence {idx + 1}/{len(sentences)}")
    idx += 1
    semantic_tokens = generate_text_semantic(
        sentence,
        history_prompt=SPEAKER,
        temp=GEN_TEMP,
        min_eos_p=0.05,  # this controls how likely the generation is to end
    )

    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]


sf.write("out/audio.wav", np.concatenate(pieces), SAMPLE_RATE)