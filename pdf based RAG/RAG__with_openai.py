# RAG with system Groq api key
import hashlib
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_INDEX_DIR = BASE_DIR / "pdf_faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o-mini"


def extract_text(pdf_docs) -> str:
    full_text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + "\n"
    return full_text


def split_text(text: str, chunk_size: int = 100, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text.strip())
    return chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_pdf_signature(pdf_docs) -> str:
    digest = hashlib.sha256()
    for pdf in pdf_docs:
        digest.update(pdf.name.encode("utf-8", errors="ignore"))
        digest.update(pdf.getvalue())
    return digest.hexdigest()[:20]


def index_path(signature: str) -> Path:
    return PDF_INDEX_DIR / signature


def build_or_load_vectorstore(pdf_docs) -> tuple[FAISS, list[str]]:
    signature = build_pdf_signature(pdf_docs)
    store_path = index_path(signature)
    embeddings = get_embedding_model()

    if store_path.exists():
        vector_store = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        chunks = [doc.page_content for doc in vector_store.docstore._dict.values()]
        return vector_store, chunks

    text = extract_text(pdf_docs)
    chunks = split_text(text)
    if not chunks:
        raise ValueError("No readable text found in the uploaded PDFs.")

    PDF_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(str(store_path))
    return vector_store, chunks


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in environment/.env")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.2,
        api_key=api_key,
    )

def get_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


def build_chain(vector_store: FAISS, top_k: int = 3) -> ConversationalRetrievalChain:
    llm = get_llm()
    memory = get_memory()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain


def app() -> None:
    st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="centered")
    st.title("Chat With PDFs (LangChain + Groq)")
    st.caption("Recursive splitting, conversation memory, and cached FAISS index.")

    if "rag_ready" not in st.session_state:
        st.session_state.rag_ready = False
        st.session_state.vector_store = None
        st.session_state.chain = None
        st.session_state.chunks = []
        st.session_state.messages = []

    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    if st.button("Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Building/loading index..."):
                vector_store, chunks = build_or_load_vectorstore(pdf_docs)
                chain = build_chain(vector_store, top_k=3)
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain
            st.session_state.chunks = chunks
            st.session_state.rag_ready = True
            st.success(f"Ready. Indexed {len(chunks)} chunks.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved Chunks"):
                    for src in msg["sources"]:
                        st.write(src)

    user_query = st.chat_input("Ask your question from uploaded PDFs...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            if not st.session_state.rag_ready:
                st.error("Please upload and process PDFs first.")
                return
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({"question": user_query})
                answer = response.get("answer", "")
                source_docs = response.get("source_documents", [])
                sources = [doc.page_content for doc in source_docs]
            st.write(answer)
            with st.expander("Retrieved Chunks"):
                for src in sources:
                    st.write(src)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    app()
