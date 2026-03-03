import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------- UI ----------------
st.title("Local RAG AI Assistant (Mahabharata)")

# ---------------- Embeddings & DB ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------- LLM ----------------
llm = Ollama(model="llama3")

# ---------------- Prompt ----------------
prompt = PromptTemplate.from_template(
    """
Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# ---------------- Helper ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------- RAG Chain ----------------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ---------------- UI Interaction ----------------
question = st.text_input("Ask a question from your documents:")

if question:
    answer = rag_chain.invoke(question)
    st.write(answer)
