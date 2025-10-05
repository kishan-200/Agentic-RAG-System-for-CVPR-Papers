This folder provides an end-to-end workflow to:

Generate and preprocess a CVPR-focused instruction dataset

Fine-tune Gemma 3 4B-IT with QLoRA, merge adapters, and export to GGUF

Build a Neo4j knowledge graph of papers (papers, authors, keywords)

Run a local RAG assistant over the paper corpus using FAISS + Ollama

The folder is organized around four main notebooks:

synthetic_dataset_generation.ipynb

Preprocessing.ipynb

finetuning.ipynb

neo4j.ipynb

RAGproject.ipynb

Follow the steps below in order.

1) Synthetic Dataset Generation
Notebook: synthetic_dataset_generation.ipynb

Purpose: Create high-quality instruction-style Q&A pairs from CVPR paper chunks using a local Ollama model.

Inputs:

cvpr_papers_chunks.jsonl with entries like {"page_content": "..."}

Outputs:

cvpr_finetuning_dataset.jsonl with lines containing:

instruction: the question

context: the source snippet

response: concise answer supported only by the context

How it works:

Loads chunks, samples up to NUM_EXAMPLES_TO_GENERATE

Prompts Ollama (e.g., llama3) in JSON mode to return strictly valid JSON

Skips malformed outputs and logs warnings

Saves valid examples to JSONL, one per line

Local requirements:

Install and run Ollama; pull the model (e.g., ollama pull llama3)

Python deps: tqdm; json is stdlib

Tips:

Ensure chunks have enough tokens (filter out very short contexts)

Manually spot-check the resulting dataset for formatting and faithfulness

2) Preprocessing
Notebook: Preprocessing.ipynb

Purpose: Clean and format the generated dataset for supervised fine-tuning.

Typical steps:

Load cvpr_finetuning_dataset.jsonl

Validate fields and JSON lines

Filter low-quality or malformed entries

Optionally deduplicate and length-filter

Save as cvpr_finetuning_formatted.jsonl

Output consumed by the fine-tuning notebook.

3) Fine-Tuning with QLoRA, Merge, and GGUF
Notebook: finetuning.ipynb

What it does:

Installs dependencies (torch, transformers, peft, datasets, bitsandbytes, accelerate, trl)

Authenticates with Hugging Face (prefer hf auth login)

Loads google/gemma-3-4b-it in 4-bit (nf4) with bitsandbytes

Runs SFT with TRL + PEFT (LoRA) on cvpr_finetuning_formatted.jsonl

Saves LoRA adapters (e.g., gemma-3-4b-cvpr)

Merges adapters with base model (merge_and_unload) and saves a merged HF model

Builds llama.cpp and converts the merged model to GGUF for local inference

Key training config (adjust for VRAM):

LoRA target modules: ["q_proj","k_proj","v_proj","o_proj"]

LoRA: r=64, alpha=16, dropout=0.1, bias="none"

4-bit quantization: nf4, compute dtype float16

Batch: per_device_train_batch_size=1, gradient_accumulation_steps=4

Memory: gradient_checkpointing=True

LR/schedule: 2e-4, cosine; warmup_ratio=0.03

Logging/saving: logging_steps=5, save_steps=25

Merging:

Load base model fp16 + tokenizer

Load adapter with PeftModel, call merge_and_unload()

Save merged model and tokenizer (e.g., to Google Drive)

GGUF conversion:

git clone llama.cpp

Build (CPU example): cmake .. -G Ninja -DLLAMA_BUILD_SERVER=ON -DLLAMA_CUDA=OFF && ninja

pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

Convert:

python3 llama.cpp/convert_hf_to_gguf.py -m /path/to/merged -o /path/to/out.gguf --outtype q4_K

Choose quant type by device needs (q4_K balanced, q6_K higher quality, q2_K smaller)

4) Neo4j Graph Ingestion
Notebook: neo4j.ipynb

Purpose: Build a knowledge graph of papers, authors, and keywords.

Configuration:

NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

JSONL_FILE_PATH = "processed_data/cvpr_papers_cleaned.jsonl"

Data model:

(Paper {title})

(Author {name})

(Keyword {term})

Relationships:

(Author)-[:AUTHORED]->(Paper)

(Paper)-[:MENTIONS]->(Keyword)

Keyword extraction:

Simple heuristic from summary text: tokens > 5 chars excluding common stop terms, up to 5 unique keywords

Replace with NLP-based keywording if needed (e.g., YAKE, KeyBERT)

Usage:

Start Neo4j (local or AuraDB)

Update credentials and path

Run the notebook; it MERGEs nodes/relationships to avoid duplicates

Console prints ingestion progress for each paper

5) RAG Assistant (FAISS + Ollama)
Notebook: RAGproject.ipynb

Purpose: Query the paper corpus with a local LLM using retrieval-augmented generation.

Configuration:

FAISS_INDEX_PATH = "faiss_cvpr_index" (prebuilt FAISS store)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOCAL_OLLAMA_MODEL = "cvpr-gemma-3-4b" (or your merged GGUF served through Ollama)

TOP_K = 5

MAX_TOKENS = 512

What it does:

Loads FAISS index with HuggingFaceEmbeddings

Connects to local Ollama model

Uses a concise CVPR QA prompt with context and question

Returns answer, latency, retrieved count, and source snippets (with paper titles where available)

Interactive loop: type questions, or 'exit' to quit

Notes:

LangChain classes (HuggingFaceEmbeddings, Ollama) have deprecation warnings; consider migrating to langchain_huggingface and langchain_ollama packages

Ensure FAISS index exists at the given path; build it in a separate step if needed

Recommended Project Structure
data/

cvpr_papers_chunks.jsonl

cvpr_finetuning_dataset.jsonl

cvpr_finetuning_formatted.jsonl

processed_data/cvpr_papers_cleaned.jsonl

vectorstore/

faiss_cvpr_index/ (FAISS files)

adapters/

gemma-3-4b-cvpr/

merged/

gemma-3-4b-cvpr-merged/ (HF model files)

gguf/

gemma3-4b-cvpr-q4k.gguf

notebooks/

synthetic_dataset_generation.ipynb

Preprocessing.ipynb

finetuning.ipynb

neo4j.ipynb

RAGproject.ipynb