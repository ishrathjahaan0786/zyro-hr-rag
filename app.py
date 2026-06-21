import os
import zipfile
import shutil
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="Zyro HR AI Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 Zyro HR AI Assistant")
st.caption("Advanced RAG Assistant powered by FAISS + Groq + LangChain")


# -----------------------------
# FIX FAISS ZIP
# -----------------------------

def prepare_faiss():

    if os.path.exists("zyro_faiss_index/index.faiss"):
        return True


    if os.path.exists("zyro_faiss_index.zip"):

        with zipfile.ZipFile(
            "zyro_faiss_index.zip",
            "r"
        ) as zip_ref:

            zip_ref.extractall("temp")


        # normal zip
        if os.path.exists(
            "temp/index.faiss"
        ):

            os.makedirs(
                "zyro_faiss_index",
                exist_ok=True
            )

            shutil.move(
                "temp/index.faiss",
                "zyro_faiss_index/index.faiss"
            )

            shutil.move(
                "temp/index.pkl",
                "zyro_faiss_index/index.pkl"
            )


        # folder inside zip
        elif os.path.exists(
            "temp/zyro_faiss_index"
        ):

            if os.path.exists("zyro_faiss_index"):
                shutil.rmtree("zyro_faiss_index")

            shutil.move(
                "temp/zyro_faiss_index",
                "zyro_faiss_index"
            )


        shutil.rmtree(
            "temp",
            ignore_errors=True
        )

    return os.path.exists(
        "zyro_faiss_index/index.faiss"
    )


# -----------------------------
# LOAD RAG SYSTEM
# -----------------------------

@st.cache_resource
def load_system():


    ready = prepare_faiss()


    if not ready:
        st.error(
            "FAISS index not found. Upload zyro_faiss_index.zip"
        )
        st.stop()



    embeddings = HuggingFaceEmbeddings(

        model_name=
        "BAAI/bge-base-en-v1.5",

        model_kwargs={
            "device":"cpu"
        },

        encode_kwargs={
            "normalize_embeddings":True
        }
    )


    db = FAISS.load_local(

        "zyro_faiss_index",

        embeddings,

        allow_dangerous_deserialization=True

    )



    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )


    if api_key is None:

        st.error(
            "Please add GROQ_API_KEY in HuggingFace Secrets"
        )

        st.stop()



    llm = ChatGroq(

        groq_api_key=api_key,

        model_name="llama-3.1-8b-instant",

        temperature=0

    )



    qa = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=db.as_retriever(
            search_kwargs={
                "k":3
            }
        ),

        return_source_documents=True

    )


    return qa



qa = load_system()


st.success(
    "🔥 HR Knowledge Base Loaded Successfully"
)


# -----------------------------
# USER INTERFACE
# -----------------------------


question = st.text_input(

    "Ask your HR question 👇",

    placeholder=
    "Example: Can employees work from home permanently?"

)



if question:


    with st.spinner(
        "Thinking..."
    ):


        result = qa.invoke(
            {
                "query": question
            }
        )


    st.subheader(
        "💡 Answer"
    )


    st.write(
        result["result"]
    )



    st.subheader(
        "📚 Sources"
    )


    for doc in result["source_documents"]:


        source = doc.metadata.get(
            "source",
            "Unknown"
        )


        page = doc.metadata.get(
            "page",
            "?"
        )


        st.write(
            f"📄 {source} - Page {page}"
        )