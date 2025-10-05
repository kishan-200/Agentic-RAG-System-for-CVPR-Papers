# Qpai_assignment

# Data pipeline:

Collected 50 CVPR-related arXiv papers (metadata, abstracts, and full text where available), performed cleaning, tokenization, and normalization.

Generated embeddings and indexed documents using FAISS/ChromaDB for efficient semantic search.

# Model fine-tuning and local hosting:

Fine-tuned Gemma 3–4B with QLoRA on domain-specific prompts and tasks.

Converted the model to GGUF(GPT-Generated Unified Format) via llama.cpp and hosted locally with Ollama for fast, resource-efficient inference on CPU/GPU.

# RAG pipeline:

Implemented a retriever-generator flow with prompt templates optimized for relevance and technical clarity.

Tuned chunking, top-k(here we considered k=5), and reranking parameters to balance accuracy and latency.

# Langflow integration:

Lanflow version 1.3.4

Built visual workflows to query, summarize, and compare papers with step-wise reasoning by using component tools available in langflow.

# Agentic capabilities:

Implemented agents to explain methodologies/results, compare multiple papers, simplify jargon for learners, and automate multi-step Q&A tasks.

# Graph visualization (Neo4j):

Created Neo4j graphs to visualize author collaboration networks, keyword/topic relationships, and citation-style linkages for exploratory analysis.

