This folder contains end-to-end notebooks and scripts to:

Generate a synthetic, instruction-style dataset from CVPR paper chunks

Fine-tune Gemma 3 4B-IT using QLoRA (SFT)

Merge LoRA adapters into the base model

Convert the merged model to GGUF for use with llama.cpp

The workflow is organized into three notebooks:

synthetic_dataset_generation.ipynb

Preprocessing.ipynb

finetuning.ipynb

Each notebook is designed to run in order.

Prerequisites
Google Colab (recommended) or local machine with NVIDIA GPU

Python 3.10+ (local for dataset generation script)

Sufficient disk space (model downloads are multiple GB)

Hugging Face account and access token to log in (optional for public models)

For GGUF conversion: building llama.cpp

Files Overview
synthetic_dataset_generation.ipynb

Generates instruction-following Q&A pairs from CVPR paper chunks using a local Ollama model (e.g., llama3).

Input: cvpr_papers_chunks.jsonl (JSONL with {"page_content": "..."}).

Output: cvpr_finetuning_dataset.jsonl (JSONL with keys: instruction, context, response).

Preprocessing.ipynb

Cleans and formats generated data into a final training JSONL file usable by HF datasets and TRL SFTTrainer.

Output: cvpr_finetuning_formatted.jsonl (expected by finetuning.ipynb).

finetuning.ipynb

Installs dependencies, logs into Hugging Face, fine-tunes google/gemma-3-4b-it with QLoRA, saves the adapter, merges adapter with base model, zips artifacts, and converts to GGUF via llama.cpp.

Pipeline
Synthetic dataset generation (local or Colab)

Open synthetic_dataset_generation.ipynb.

Configure:

INPUT_CHUNKS_FILE: path to cvpr_papers_chunks.jsonl

OUTPUT_DATASET_FILE: path to save cvpr_finetuning_dataset.jsonl

NUM_EXAMPLES_TO_GENERATE: number of samples (e.g., 200)

OLLAMA_MODEL: local model name (e.g., llama3)

Requirements (local):

Install Ollama and pull the model:

ollama pull llama3

Python dependencies: tqdm, json (stdlib)

Run the notebook:

It will iterate over chunks, prompt the model to produce valid JSON objects:

Keys: "instruction", "context", "response"

Invalid JSON outputs are skipped with a warning.

Output:

cvpr_finetuning_dataset.jsonl

Manual review:

Inspect random samples to ensure correctness, technicality, and answerability strictly from context.

Preprocess to training format

Open Preprocessing.ipynb.

Load cvpr_finetuning_dataset.jsonl and clean/validate:

Remove malformed lines

Normalize fields

Optionally filter length, deduplicate, etc.

Save as:

cvpr_finetuning_formatted.jsonl

This file is referenced in finetuning.ipynb via dataset_path.

Fine-tuning with QLoRA (Colab recommended)

Open finetuning.ipynb in Colab.

Step 1: Install dependencies

torch, transformers, peft, datasets, bitsandbytes, accelerate, trl

Step 2: Login to Hugging Face

Use hf auth login or huggingface-cli login (deprecated message suggests hf auth login)

Provide a valid token with write permissions if planning to push artifacts

Step 3: QLoRA fine-tune

Model: google/gemma-3-4b-it

Dataset: cvpr_finetuning_formatted.jsonl

LoRA target modules: ["q_proj","k_proj","v_proj","o_proj"]

LoRA config: r=64, alpha=16, dropout=0.1

4-bit quantization: nf4, compute dtype float16

TrainingArguments tuned for ~16GB VRAM (T4):

per_device_train_batch_size=1

gradient_accumulation_steps=4

gradient_checkpointing=True

learning_rate=2e-4

warmup_ratio=0.03

cosine scheduler

save_steps=25, logging_steps=5

Trainer: TRL SFTTrainer with peft_config

Output:

Adapter saved to new_model_name (e.g., gemma-3-4b-cvpr)

Step 4: Merge adapter with base model

Load base model (fp16), load PeftModel from adapter, call merge_and_unload()

Save merged model and tokenizer to merged_model_path (e.g., Google Drive)

Optional: zip adapters to download for backup

Step 5: Convert merged model to GGUF

apt-get update and install build-essential, cmake, git, g++, make

git clone https://github.com/ggerganov/llama.cpp

Build:

cd llama.cpp && mkdir -p build && cd build

cmake .. -G Ninja -DLLAMA_BUILD_SERVER=ON -DLLAMA_CUDA=OFF

ninja

Install conversion requirements:

pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

Run conversion script (from llama.cpp):

python3 convert_hf_to_gguf.py -m /path/to/merged-model -o /path/to/output/gguf --outtype q4_K or desired quant

Result:

A .gguf file suitable for llama.cpp runtime

Note:

For GPU inference in llama.cpp, build with -DLLAMA_CUDA=ON and ensure CUDA toolchain availability.