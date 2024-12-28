# Environment Setup
If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```
Otherwise, while installing it will build the llama.cpp x86 version which will be **10x** slower than the arm64 version.

Then, create a new conda environment:
```bash
conda create -n audio_control python=3.8
conda activate audio_control
```
## Install the required packages
Install the Hugging Face CLI and the llama-cpp-python package:
```bash
pip install huggingface-hub 
CMAKE_ARGS="-DGGML_METAL=on" 
pip install llama-cpp-python
```
Please refer to the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) if there are any issues with the installation.
Install Whisper.cpp Python bindings package
```bash
pip install git+https://github.com/stlukey/whispercpp.py
```
If you have an older version of Numpy, you may need to upgrade it to support the Whisper.cpp package:
```bash
pip install numpy --upgrade
```
Please refer to the [whispercpp.py](https://github.com/stlukey/whispercpp.py) if there are any issues with the installation.
### Build issues
The following error may occur while building the llama-cpp-python package:

```bash 
llvm-ar: adding 50 object files to ...
error: Command "/Users/runner/miniforge3/conda-bld/python-split_1606376626618/_build_env/bin/llvm-ar  ...  failed with exit status 127
```
To fix this, you can set the AR environment variable to the system ar:
```bash
export AR=/usr/bin/ar
```

## Downloading .gguf models from Hugging Face
To use the Llama (or any other) model, you need to download quantized version of the model from the Hugging Face model hub.
Create model directories:
```bash
mkdir -p models/small
mkdir -p models/medium
```

Download the models:
```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
    mistral-7b-instruct-v0.1.Q4_K_S.gguf \
    --local-dir models/small
```
This will download 4-bit quantized small version of Mistral 7B model. You can also download the medium version. 

Refer to the [Hugging Face model hub](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF#:~:text=of%20quantisation%20methods-,Click,-to%20see%20details) for more information on GGUF models.
The more RAM you have, the larger model you can use. For 16 GB RAM on M1 Mac, it is recommended to use the small model or the medium model (both quantized to 4-bit).

## Running Audio Assistant
If you have an OpenAI API key, please set it as an environment variable before running the script(optional):
```bash
export OPENAI_API_KEY=your-api-key
```
Then you can run the script with the following command:
```bash
python spot_assistant.py --inference-method local --llama-model ./models/7B/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```
The script will let you start and stop an audio by pressing "Enter". It will first transcribe the audio using local Whisper or OpenAI API Whisper, then it will generate appropriate [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) using local Llama model or OpenAI API GPT4o-mini model which is the most cost effective model currently available.

Overall, it seems like local Whisper transcription is quite fast, but Llama (medium) can be slow. It is worth trying the small model first or using the OpenAI API for the inference.

