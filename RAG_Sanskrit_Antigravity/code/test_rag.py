import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time

# Configuration
DB_PATH = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def main():
    print("--- Starting RAG Test Script ---")
    
    # 1. Check Vector DB
    if not os.path.exists(DB_PATH):
        print(f"Error: Vector DB not found at {DB_PATH}. Run ingest.py first.")
        return

    # 2. Check Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Run download_model.py first.")
        return

    print("Loading LLM...")
    try:
        llm = CTransformers(
            model=MODEL_PATH,
            model_type="mistral",
            config={'max_new_tokens': 256, 'temperature': 0.1, 'context_length': 2048}
        )
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return

    print("Loading Retriever...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 3. Setup Chain
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 4. Run Query (Replace with a relevant question from Rag-docs.docx content if known, otherwise generic)
    # Assuming the doc contains something about Sanskrit/ Vedas / or the specific content of Rag-docs.
    # I'll ask a generic summarization question or strictly prompt-based.
    # Since I don't know the exact content of Rag-docs.docx, I'll ask "What is the main topic of the document?"
    query = "Summarize the content of the provided documents."
    
    print(f"\nQuery: {query}")
    print("Thinking...")
    
    start_time = time.time()
    try:
        response = qa_chain.invoke({"query": query})
        end_time = time.time()
        latency = end_time - start_time
        
        print("\n--- Answer ---")
        print(response["result"])
        print(f"\nLatency: {latency:.2f} seconds")
        
        print("\n--- Sources ---")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
