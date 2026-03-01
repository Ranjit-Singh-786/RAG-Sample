import os
from pathlib import Path
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "common_question.csv"
INDEX_PATH = BASE_DIR / "faiss.index"

# ==========================
# 1. Load CSV
# ==========================
def load_questions_csv(csv_path: Path) -> pd.DataFrame:
    # Try common encodings because CSV may be saved from Excel/Windows.
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


df = load_questions_csv(CSV_PATH)

df["combined"] = "Question: " + df["prompt"] + "\nAnswer: " + df["response"]
documents = df["combined"].tolist()

# ==========================
# 2. Embedding Model
# ==========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(docs: list[str]) -> faiss.Index:
    embeddings = embedding_model.encode(docs)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dimension)
    new_index.add(embeddings)
    faiss.write_index(new_index, str(INDEX_PATH))
    print("FAISS index created with", new_index.ntotal, "documents")
    return new_index


# ==========================
# 3. FAISS Index (Cosine)
# ==========================
if INDEX_PATH.exists():
    index = faiss.read_index(str(INDEX_PATH))
    if index.ntotal != len(documents):
        print("Index and CSV row count mismatch. Rebuilding index...")
        index = build_index(documents)
    else:
        print("Loaded existing FAISS index with", index.ntotal, "documents")
else:
    index = build_index(documents)

# ==========================
# 4. Groq Client Setup
# ==========================
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY missing in environment/.env")
client = Groq(api_key=api_key)

# ==========================
# 5. RAG Function
# ==========================
def ask_question(user_query, top_k=3):

    query_embedding = embedding_model.encode([user_query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Retrieve similar documents
    distances, indices = index.search(query_embedding, top_k)

    retrieved_docs = [documents[i] for i in indices[0]]
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

    # Call Groq LLM
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",  # Recommended fast model
        temperature=0.2
    )

    return chat_completion.choices[0].message.content


# ==========================
# 6. Test Loop
# ==========================
if __name__ == "__main__":
    while True:
        query = input("\nAsk your question: ")
        if query.strip().lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        answer = ask_question(query)
        print("\nBot:", answer)
