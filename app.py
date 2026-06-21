import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Zyro HR AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🏢 Zyro Dynamics HR Intelligence")
st.caption("AI powered HR assistant using RAG + Groq + LangChain")

@st.cache_resource
def load_system():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.load_local(
        "zyro_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    return db, llm

vectorstore, llm = load_system()

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)

question = st.chat_input("Ask your HR question...")

if question:
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    docs = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are Zyro Dynamics HR Policy Expert.

Use only this context.
If unavailable say Information Not Available.

Context:
{context}

Question:
{question}

Answer with:
Decision:
Policy Explanation:
"""

    response = llm.invoke(prompt).content

    sources = set()
    for d in docs[:3]:
        sources.add(d.metadata["source"].split("/")[-1])

    response += "\n\n📚 Sources:\n"
    for s in sources:
        response += "- " + s + "\n"

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat.append(("assistant", response))
