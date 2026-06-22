# 🤖 LaibaBot – Personal Academic Assistant

## Natural Language Processing (CC438) Semester Project

### Student Information

**Name:** Laiba Ishaq
**Program:** BS Computer Science (BSCS)
**University:** University of Management and Technology (UMT), Lahore
**Course:** Natural Language Processing (CC438)

---

# Project Overview

LaibaBot is a Retrieval-Augmented Generation (RAG) based chatbot developed as a semester project for the Natural Language Processing course.

The chatbot is designed to answer questions related to the student's academic profile, skills, projects, and coursework by retrieving information from personal documents and generating context-aware responses using a Large Language Model (LLM).

The system combines document retrieval with generative AI to provide accurate and relevant answers based on the uploaded dataset.

---

# Project Objectives

* Build a personalized chatbot with a unique identity.
* Implement a Retrieval-Augmented Generation (RAG) pipeline.
* Use personal/custom documents as the knowledge base.
* Generate contextual responses using an LLM.
* Maintain conversation history.
* Deploy the chatbot online for public access.

---

# Chatbot Identity

**Chatbot Name:** LaibaBot

LaibaBot acts as a personal academic assistant capable of answering questions about:

* Educational background
* Technical skills
* Academic projects
* HCI project findings
* Semester work
* Personal profile information

---

# Dataset

The chatbot uses custom personal documents:

1. CV.pdf
2. HCI_Project.pdf
3. NLP_Project.pdf

These documents are loaded, processed, embedded, and indexed for retrieval.

---

# System Architecture

### Step 1: Document Loading

PDF documents are loaded using LangChain document loaders.

### Step 2: Text Preprocessing

Documents are divided into smaller chunks using Recursive Character Text Splitter.

### Step 3: Embedding Generation

Text chunks are converted into vector embeddings using Hugging Face Sentence Transformers.

### Step 4: Vector Database

Embeddings are stored in a FAISS vector database.

### Step 5: Retrieval

Relevant document chunks are retrieved based on user queries.

### Step 6: Response Generation

Retrieved context is provided to Google's Gemini LLM to generate accurate responses.

### Step 7: Conversation Memory

Chat history is maintained using LangChain memory components.

---

# Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Embeddings
* Google Gemini API
* PyPDF
* Dotenv

---

# Project Structure

LaibaBot/

├── app.py

├── requirements.txt

├── README.md

├── .env

└── data/

    ├── CV.pdf

    ├── HCI_Project.pdf

    └── NLP_Project.pdf

---

# Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a .env file and add:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

# Running the Project

Start the Streamlit application:

```bash
streamlit run app.py
```

---

# Example Queries

* Who is Laiba Ishaq?
* What skills does Laiba have?
* Tell me about Laiba's projects.
* What is the HCI project about?
* What usability principles were identified in Fiverr?
* What is Laiba's educational background?

---

# Features

* Personalized chatbot identity
* PDF document processing
* Retrieval-Augmented Generation (RAG)
* FAISS vector search
* Context-aware responses
* Conversation history
* Streamlit web interface
* Online deployment support

---

# Future Improvements

* Multi-document upload support
* Voice interaction
* Agentic RAG functionality
* Advanced memory management
* Improved UI/UX design

---

# Author

Laiba Ishaq

BS Computer Science

University of Management and Technology (UMT)

---

# License

This project is developed solely for academic and educational purposes as part of the Natural Language Processing course.
