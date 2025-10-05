This folder contains a set of Langflow JSON flows to build a local, privacy-preserving research assistant for computer vision papers. The system combines FAISS-based retrieval over a curated CVPR index, Ollama-hosted LLMs, and Hugging Face TEI embeddings to enable fast query, comparison, summarization, and agentic explanations with memory.

Contents:

Query-Research-Papers.json
Summarize.json
Support-Step-based-reasoning-Langchain.json
Memory-Langraph-agentic-AI.json

Each file is an exportable Langflow graph you can import directly into Langflow.


2. High-level Architecture:

Vector store: FAISS index on disk at E:/ollama_model/faiss_cvpr_index, index_name=index
Embeddings: Hugging Face TEI endpoint (default http://localhost:8080/) using sentence-transformers/all-MiniLM-L6-v2
LLM: Ollama local models (default base_url http://localhost:11434), e.g., cvpr-gemma-3-4b:latest, llama3, gemma3
Retrieval: LangChain RetrievalQA with return_source_documents enabled internally for traceability
Tools: Agentic flow exposes tools for research, memory read/write, and optional “current date” tool
Memory: Optional external memory integration; default uses Langflow’s internal tables

3. Prerequisites:

Langflow 1.3.4+
Python 3.10+

4. Ollama running locally with required models pulled:

cvpr-gemma-3-4b:latest
cvpr-assistant-meta:latest (optional)
llama3:latest or llama3.2:3b
gemma3:latest

5. Hugging Face Text Embeddings Inference (TEI) running locally:

Endpoint: http://localhost:8080/
Model: sentence-transformers/all-MiniLM-L6-v2

6. A local FAISS index directory populated with your paper corpus:

Path: E:/ollama_model/faiss_cvpr_index
Files should be named index.faiss and index.pkl (LangChain FAISS format)


7. Quick Start:

Start services
Launch TEI at http://localhost:8080/ with the MiniLM model.
Start Ollama at http://localhost:11434 and pull models:
ollama pull cvpr-gemma-3-4b:latest
ollama pull llama3:latest (or llama3.2:3b)
ollama pull gemma3:latest
Open Langflow
Run Langflow and import each JSON via the UI (Import Flow).

8. Configure nodes:-

In HuggingFace Embeddings Inference nodes:

inference_endpoint: http://localhost:8080/
model_name: sentence-transformers/all-MiniLM-L6-v2

In FAISS Loader & Retriever nodes:

persist_directory: E:/ollama_model/faiss_cvpr_index
index_name: index
k: set desired top-k (4–8 typical)

In Ollama nodes:
base_url: http://localhost:11434
model_name: choose one of the pulled models
temperature: 0.1 by default for faithful answers

9. Run a flow:

Click the Chat Input/Text Input nodes, provide a query or topics, and execute the flow to view results in Chat Output.