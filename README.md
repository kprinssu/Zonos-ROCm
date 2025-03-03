# Zonos-v0.1-ROCm

<div align="center">
<img src="assets/ZonosHeader.png" 
     alt="Alt text" 
     style="width: 500px;
            height: auto;
            object-position: center top;">
</div>

<div align="center">
  <a href="https://discord.gg/gTW9JwST8q" target="_blank">
    <img src="https://img.shields.io/badge/Join%20Our%20Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

---

Zonos-v0.1 is a leading open-weight text-to-speech model trained on more than 200k hours of varied multilingual speech, delivering expressiveness and quality on par with—or even surpassing—top TTS providers.

Our model enables highly natural speech generation from text prompts when given a speaker embedding or audio prefix, and can accurately perform speech cloning when given a reference clip spanning just a few seconds. The conditioning setup also allows for fine control over speaking rate, pitch variation, audio quality, and emotions such as happiness, fear, sadness, and anger. The model outputs speech natively at 44kHz.

##### For more details and speech samples, check out Zyphra's blog [here](https://www.zyphra.com/post/beta-release-of-zonos-v0-1)

---

Zonos follows a straightforward architecture: text normalization and phonemization via eSpeak, followed by DAC token prediction through a transformer or hybrid backbone. An overview of the architecture can be seen below.

<div align="center">
<img src="assets/ArchitectureDiagram.png" 
     alt="Alt text" 
     style="width: 1000px;
            height: auto;
            object-position: center top;">
</div>

---


## Features

- Zero-shot TTS with voice cloning: Input desired text and a 10-30s speaker sample to generate high quality TTS output
- Audio prefix inputs: Add text plus an audio prefix for even richer speaker matching. Audio prefixes can be used to elicit behaviours such as whispering which can otherwise be challenging to replicate when cloning from speaker embeddings
- Multilingual support: Zonos-v0.1 supports English, Japanese, Chinese, French, and German
- Audio quality and emotion control: Zonos offers fine-grained control of many aspects of the generated audio. These include speaking rate, pitch, maximum frequency, audio quality, and various emotions such as happiness, anger, sadness, and fear.
- Fast: our model runs with a real-time factor of ~2x on an RTX 4090 (i.e. generates 2 seconds of audio per 1 second of compute time)
- Gradio WebUI: Zonos comes packaged with an easy to use gradio interface to generate speech
- Simple installation and deployment: Zonos can be installed and deployed simply using the docker file packaged with our repository.

## Installation
#### System requirements
- **Operating System:** Linux (preferably Ubuntu 22.04/24.04)
- **ROCm Version** Requires ROCm version 6.2+ installed [AMD Install Instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 
	```
	amdgpu-install --usecase=rocm,lrt,hiplibsdk,mllib,mlsdk 
	# May need to add --no-32 (Only use this option if you're getting errors and know what you're doing)
	```
- **GPU:** 6GB+ VRAM, Hybrid-model requires a RDNA2-series (RX 6000+) or newer AMD GPU

#### System dependencies

Zonos depends on the eSpeak library phonemization. You can install it on Ubuntu with the following command:

```bash
sudo apt install espeak-ng # For Ubuntu
```

##### Installing using pip

```bash
git clone https://github.com/YellowRoseCx/Zonos-ROCm
git clone https://github.com/YellowRoseCx/flash-attention-triton
git clone https://github.com/state-spaces/mamba
git clone https://github.com/Dao-AILab/causal-conv1d

cd Zonos
python -m venv venv
source zonos/venv/bin/activate
pip install -r requirements-rocm.txt

cd ../causal-conv1d
pip install .

cd ../mamba
pip install . --no-build-isolation

cd ../flash-attention
pip uninstall triton
rm -rf ~/.triton
rm -rf ./triton/python/build

git clone https://github.com/triton-lang/triton
cd triton
pip install --verbose --no-build-isolation ./python
cd ..

pip show triton

export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
python setup.py install

cd ../Zonos-ROCm
pip install -e .
pip install --no-build-isolation -e .[compile] # optional but needed to run the hybrid
```

##### Confirm that it's working

For convenience we provide a minimal example to check that the installation works:

```
python sample.py
```

## Usage

### Gradio interface (recommended)

```bash
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export TORCH_BLAS_PREFER_HIPBLASLT=0
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1 

export PYTORCH_TUNABLEOP_VERBOSE=1 
export PYTORCH_TUNABLEOP_TUNING=1 
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME=~/tunableop-config-mamba.csv 

export MIOPEN_FIND_ENFORCE=3 
export MIOPEN_FIND_MODE=5 

python gradio_interface.py
```

### Python

```python
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
```


This should produce a `sample.wav` file in your project root directory.

_For repeated sampling we highly recommend using the gradio interface instead, as the minimal example needs to load the model every time it is run._