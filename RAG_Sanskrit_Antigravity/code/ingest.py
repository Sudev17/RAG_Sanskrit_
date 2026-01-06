import os
import argparse
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Constants
DATA_PATH = "data"
DB_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_documents(directory):
    documents = []
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
    return documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database")
    args = parser.parse_args()

    if args.reset and os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)
        print(f"Cleared database at {DB_PATH}")

    print(f"Loading documents from {DATA_PATH}...")
    docs = load_documents(DATA_PATH)
    print(f"Loaded {len(docs)} documents.")

    if not docs:
        print("No documents found. Exiting.")
        return

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Creating embeddings using {EMBEDDING_MODEL}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Indexing into ChromaDB at {DB_PATH}...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
