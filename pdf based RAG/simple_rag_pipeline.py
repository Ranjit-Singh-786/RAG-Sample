import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # to  prevent duplicate OpenMP error
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ PDF TEXT ------------------
def extract_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


# ------------------ SPLIT ------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# ------------------ VECTOR STORE ------------------
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(chunks, embeddings)


# ------------------ SIMPLE RAG ------------------
def ask_question(vector_store, query):
    docs = vector_store.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided context. If not found, say 'Not found in document'."
            },
            {
                "role": "user",
                "content":f"""
Context:
{context}

Question:
{query}
"""
            }
        ]
    )

    answer = response.choices[0].message.content
    return answer, context



def app():
    st.title("💬 PDF Chatbot (RAG System)")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.header("📄 Upload PDF")

        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("Process"):
            text = extract_text(pdf_docs)
            chunks = split_text(text)
            vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.success("PDF processed!")

    # ------------------ CHAT HISTORY ------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ------------------ USER INPUT ------------------
    query = st.chat_input("Ask something from your PDF...")

    if query:
        # user message add
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # bot response
        if st.session_state.vector_store:
            answer, context = ask_question(st.session_state.vector_store, query)
        else:
            answer = "⚠️ Please upload and process PDF first."

        # bot message add
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("📚 Context used"):
                st.write(context if st.session_state.vector_store else "No context")



if __name__ == "__main__":
    app()




