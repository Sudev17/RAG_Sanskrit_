import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Configuration
DB_PATH = "../data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

st.set_page_config(page_title="Sanskrit RAG System", layout="wide")

@st.cache_resource
def get_llm():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please run download_model.py first.")
        return None
    
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
    )
    return llm

@st.cache_resource
def get_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

def main():
    st.title("🕉️ Sanskrit Document RAG System")
    st.markdown("Query your Sanskrit documents using CPU-based local LLM.")

    # Check for Vector DB
    if not os.path.exists(DB_PATH):
        st.warning("Vector Database not found. Please run `ingest.py` first.")
        return

    llm = get_llm()
    retriever = get_retriever()
    
    if not llm or not retriever:
        return

    # Custom Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Answer in English or Sanskrit as appropriate.

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

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question in Sanskrit or English...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"_{doc.page_content[:200]}..._")

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
