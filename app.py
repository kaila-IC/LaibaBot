import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv


# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("AQ.Ab8RN6IiboAoO9jubaZ5Yz1T1QT_ccwbGOVgqejAWi8qPGPsPg")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found in .env file")
    st.stop()


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="LaibaBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LaibaBot")
st.subheader("Personal Academic Assistant")


# -----------------------------
# Load Documents
# -----------------------------
@st.cache_resource
def load_and_create_vectorstore():

    documents = []

    pdf_files = [
        "data/CV.pdf",
        "data/HCI_Project.pdf",
        "data/NLP_Project.pdf"
    ]

    for pdf in pdf_files:

        if os.path.exists(pdf):
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# -----------------------------
# Create Retriever
# -----------------------------
vectorstore = load_and_create_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)


# -----------------------------
# Gemini Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# -----------------------------
# RAG Chain
# -----------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)


# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# User Input
# -----------------------------
user_question = st.chat_input(
    "Ask something about Laiba..."
)


if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Thinking..."):

        result = qa_chain(
            {
                "question": user_question
            }
        )

        answer = result["answer"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)