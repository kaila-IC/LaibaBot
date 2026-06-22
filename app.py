import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# -----------------------------
# API KEY (Streamlit Cloud safe)
# -----------------------------
GOOGLE_API_KEY = st.secrets["AQ.Ab8RN6ICe9XGoXDuwSnbbE7opMIe9dA4Uo7c8W1wsiQU3yUh1Q"]

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="LaibaBot", page_icon="🤖")
st.title("🤖 LaibaBot")
st.write("Your Personal Academic Assistant")

# -----------------------------
# Load & Process Documents
# -----------------------------
@st.cache_resource
def create_vectorstore():

    pdf_files = [
        "data/CV.pdf",
        "data/HCI_Project.pdf",
        "data/NLP_Project.pdf"
    ]

    docs = []

    for file in pdf_files:
        if os.path.exists(file):
            loader = PyPDFLoader(file)
            docs.extend(loader.load())

    if len(docs) == 0:
        st.error("No PDFs found in data folder")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM (Gemini)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about Laiba...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):

        result = qa_chain.invoke({"question": user_input})
        answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)