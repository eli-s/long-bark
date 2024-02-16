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

script = """
Welcome to our quick dive into React Hooks - the game-changer in how we write React components!
React Hooks have been around since version 16.8, revolutionizing component development by allowing us to use state and other React features without writing a class. Today, we're going to uncover the basics of Hooks and show you how to use them. Let's get started!
Hooks offer a way to 'hook into' React features from functional components. The most commonly used Hook is useState. Let's see it in action.
Here, useState gives us two things: the current state value, count, and a function to update it, setCount. This is much simpler than the this.setState method in class components.
Next up, let's talk about useEffect. This Hook lets you perform side effects in your components, such as fetching data, directly subscribing to updates, and more.
With useEffect, you tell React to do something after render. React will remember the function you passed (we refer to it as an 'effect'), and call it later after performing the DOM updates.
And there you have it! A quick look at useState and useEffect. Hooks are powerful and flexible, making them a valuable tool in your React toolkit.
Thanks for watching! Dive into the React docs or check out more of our tutorials to get even deeper into Hooks and other advanced features. Happy coding!
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)

GEN_TEMP = 0.6
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    semantic_tokens = generate_text_semantic(
        sentence,
        history_prompt=SPEAKER,
        temp=GEN_TEMP,
        min_eos_p=0.05,  # this controls how likely the generation is to end
    )

    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]


sf.write("out/react_hooks.wav", np.concatenate(pieces), SAMPLE_RATE)