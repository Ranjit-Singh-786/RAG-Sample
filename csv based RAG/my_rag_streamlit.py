import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "common_question.csv"
INDEX_PATH = BASE_DIR / "faiss.index"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


def load_questions_csv(csv_path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "read_csv",
        b"",
        0,
        1,
        f"Unable to decode CSV: {csv_path}. Try resaving as UTF-8.",
    )


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_data(show_spinner=False)
def get_documents() -> list[str]:
    df = load_questions_csv(CSV_PATH).fillna("")
    df["combined"] = "Question: " + df["prompt"].astype(str) + "\nAnswer: " + df["response"].astype(str)
    return df["combined"].tolist()


def build_index(docs: list[str], model: SentenceTransformer) -> faiss.Index:
    embeddings = model.encode(docs)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)
    faiss.write_index(new_index, str(INDEX_PATH))
    return new_index


@st.cache_resource(show_spinner=False)
def get_index(docs_count: int) -> faiss.Index:
    if INDEX_PATH.exists():
        idx = faiss.read_index(str(INDEX_PATH))
        if idx.ntotal == docs_count:
            return idx

    model = get_embedding_model()
    docs = get_documents()
    return build_index(docs, model)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing in environment/.env")
    return Groq(api_key=api_key)


def ask_question(user_query: str, top_k: int = 3) -> tuple[str, list[str]]:
    docs = get_documents()
    model = get_embedding_model()
    index = get_index(len(docs))

    query_embedding = model.encode([user_query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    k = min(top_k, len(docs))
    _, indices = index.search(query_embedding, k)

    retrieved_docs = [docs[i] for i in indices[0] if i >= 0]
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a helpful assistant.
Answer strictly from the given context.
If the answer is not in context, say:
"I don't have information about that in my knowledge base."

Context:
{context}

User Question:
{user_query}

Answer:
"""

    client = get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.2,
    )

    return chat_completion.choices[0].message.content or "", retrieved_docs


def rebuild_index_now() -> int:
    docs = get_documents()
    model = get_embedding_model()
    build_index(docs, model)
    get_index.clear()
    return len(docs)


def app() -> None:
    st.set_page_config(page_title="Student Support RAG", page_icon="🎓", layout="centered")
    st.title("Student Support RAG Chat")
    st.caption("CSV + FAISS + Groq")

    try:
        docs = get_documents()
        _ = get_index(len(docs))
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.stop()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Rebuild Index"):
            with st.spinner("Rebuilding index..."):
                count = rebuild_index_now()
            st.success(f"Index rebuilt for {count} rows.")
    with col2:
        st.write(f"Docs in KB: **{len(docs)}**")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved Context"):
                    for s in msg["sources"]:
                        st.write(s)

    user_query = st.chat_input("Ask your question...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask_question(user_query, top_k=3)
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    return
            st.write(answer)
            with st.expander("Retrieved Context"):
                for s in sources:
                    st.write(s)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    app()
