#  ChatGPT-like PDF Q&A Web App

An AI-powered web application that allows users to chat with multiple PDF documents using a conversational interface similar to ChatGPT.

---

##  Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to provide accurate, context-aware answers from uploaded PDF documents.

Users can upload multiple PDFs, ask questions, and receive intelligent responses based on document content.

---

##  Features

-  ChatGPT-like conversational UI  
-  Upload and process multiple PDFs  
-  Context-aware answers using RAG  
-  Semantic search with vector embeddings  
-  Chat history support  
-  Fully local execution using Ollama (no API cost)  

---

##  Tech Stack

- Python  
- Streamlit  
- LangChain  
- FAISS (Vector Database)  
- Ollama (Local LLM)  

---

##  How It Works

1.  Upload PDF files  
2.  Split text into smaller chunks  
3.  Convert text into embeddings  
4.  Store embeddings in FAISS vector database  
5.  Retrieve relevant chunks based on user query  
6.  Generate answers using LLM (Ollama)  

---

##  Installation & Setup

### 1. Clone the repository

```bash
<img width="1900" height="837" alt="image" src="https://github.com/user-attachments/assets/cfeb97bf-d3c7-4381-8a70-5632991f2401" />

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
