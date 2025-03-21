# Zonos-v0.1-ROCm

## Installation
#### System requirements
- **Operating System:** Linux (preferably Ubuntu 22.04/24.04)
- **ROCm Version** Requires ROCm version 6.2+ installed [AMD Install Instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 
	```
	amdgpu-install --usecase=rocm,lrt,hiplibsdk,mllib,mlsdk 
	# May need to add --no-32 (Only use this option if you're getting errors and know what you're doing)
	```
- **GPU:** 6GB+ VRAM, Hybrid-model requires a RDNA2-series (RX 6000+) or newer AMD GPU
- Zonos depends on the eSpeak library phonemization. You can install it on Ubuntu with the following command:
```bash
sudo apt install espeak-ng # For Ubuntu
```

### Installing with FlashAttention2 and Mamba for AMD using pip

```bash
git clone https://github.com/YellowRoseCx/Zonos-ROCm
git clone https://github.com/ROCm/flash-attention -b main_perf --depth 1
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

pip install triton==3.2.0
pip show triton

export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
python setup.py install

cd ../Zonos-ROCm
pip install -e .
pip install --no-build-isolation -e .[compile] # optional but needed to run the hybrid
```

###  Increase Performance: Tuning On AMD

Considerable performance gains can be had through setting a few environment variables and letting ROCm(MIOpen) and Triton tune for the fastest GEMM kernels.
MIOpen Tuning documentation: [Using MIOPEN_FIND_ENFORCE](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/perfdb.html#using-miopen-find-enforce) and Using [https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html#find-modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html#find-modes) environment variables.
PyTorch TunableOP documentation: [TunableOP](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md)

AMD GPU users not on an RX 7000 series or higher will want to export these 2 variables:
```bash
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 
export TORCH_BLAS_PREFER_HIPBLASLT=0 
```
### Setting up for the first few runs
``export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=1`` Enable Triton autotune      
``export PYTORCH_TUNABLEOP_VERBOSE=1`` Enables Verbosity          
``export PYTORCH_TUNABLEOP_TUNING=1`` Enables Actual Tuning with PyTorch            
``export PYTORCH_TUNABLEOP_ENABLED=1`` Enable use of Tuned Results with PyTorch             
``export MIOPEN_FIND_ENFORCE=3`` Perform auto-tune. If PerfDb already contains optimized values, auto-tune is not performed.                
``export MIOPEN_FIND_MODE=1`` This is the full find mode call, which benchmarks all the solvers.             
``export PYTORCH_TUNABLEOP_FILENAME=~/tunableop-config.csv`` Typically, you would leave this variable unset, but we can use it here in order to separate results from various tuning runs.              

After setting those variables, start up Zonos ``python gradio_interface.py`` and then run through a few generations, it will take a long time(possible hours) as it finds the best solutions to the math involved in the model generation. After doing this a few times, you can CTRL+C out of the program which then saves your results.

Afterwards, set these new environment variable values which turn off tuning and sets MIOpen kernel search mode to Fast:

```bash
export PYTORCH_TUNABLEOP_TUNING=0
export MIOPEN_FIND_ENFORCE=1
export MIOPEN_FIND_MODE=2
```
Now when you rerun Zonos, it should be much quicker.

### Attention if using Hybrid model:
If using the hybrid model, you will want to comment out the following line in ``gradio_interface.py`` around line 217:
```
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")
```


``FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" FLASH_ATTENTION_TRITON_AMD_REF=1`` may be useful in the future

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

export MIOPEN_FIND_ENFORCE=1
export MIOPEN_FIND_MODE=5 

python gradio_interface.py
```

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
