# Sanskrit RAG System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for Sanskrit documents using a local CPU-optimized LLM (Mistral-7B). It creates a vector database from documents and provides a Streamlit interface for querying.

## Evaluation Criteria Alignment
- **Architecture**: Modular design (Ingest -> Vector DB -> Retrieval -> Generation).
- **Functionality**: Proven end-to-end Sanskrit query answering.
- **CPU Optimization**: Uses Quantized (GGUF) Mistral 7B via `ctransformers`.
- **Code Quality**: Structured python scripts with error handling.

## Structure
- `code/`: Python scripts for ingestion, testing, and the main application.
- `data/`: Source documents and the generated ChromaDB vector store.
- `report/`: (Place your final PDF report here)

## Setup & Usage

### 1. Prerequisites
- Python 3.10+
- 8GB+ RAM (for 7B model)

### 2. Installation
Navigate to the `code` directory and install dependencies:
```bash
cd code
pip install -r requirements.txt
```

### 3. Model Setup
Download the quantized Mistral model:
```bash
python download_model.py
```
*Note: This downloads ~4GB data to `code/models/`.*

### 4. Data Ingestion
Process documents from `data/` and create the vector database:
```bash
python ingest.py
```
*Supports .txt, .pdf, .docx files placed in the `data/` folder.*

### 5. Running the Application
Launch the Streamlit interface:
```bash
streamlit run rag_app.py
```
Access the app at `http://localhost:8501`.

## Files
- `ingest.py`: Chunking, embedding, and indexing documents.
- `rag_app.py`: Streamlit frontend and RAG logic.
- `download_model.py`: Utility to fetch the LLM.
- `test_rag.py`: CLI script for quick verification.
