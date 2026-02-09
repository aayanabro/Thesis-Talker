# 🎓 Thesis-Talker

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Inference-Groq-orange.svg)](https://groq.com/)

**Thesis-Talker** is a high-performance RAG (Retrieval-Augmented Generation) application designed to transform dense academic papers into interactive, chat-ready knowledge bases. By combining the lightning-fast inference of **Groq** with **LangChain's** orchestration, Thesis-Talker allows researchers and students to query complex documents and receive context-grounded answers in milliseconds.

---

## 🔍 Table of Contents
* [Core Features](#-core-features)
* [System Architecture](#-system-architecture)
* [Quick Start](#-quick-start)
* [Technical Details](#-technical-details)
* [Troubleshooting](#-troubleshooting)

---

## ✨ Core Features

* **Ultra-Fast Q&A:** Leverages Groq’s LPU (Language Processing Unit) to eliminate LLM latency.
* **Contextual Accuracy:** Uses a "Stuff Documents" chain to ensure the LLM only answers based on the provided PDF content.
* **Local Vector Intelligence:** Utilizes **FAISS** for similarity search and **HuggingFace** embeddings, keeping the vectorization process efficient and local.
* **Persistent Chat History:** A seamless user interface that remembers the conversation flow during your session.

---

## 🏗 System Architecture

Thesis-Talker follows a modern RAG pipeline:
1.  **Ingestion:** Extracts text from your research PDF using `PyPDFLoader`.
2.  **Chunking:** Breaks down text into 1,000-character segments with a 100-character overlap to maintain semantic continuity.
3.  **Embedding:** Converts text chunks into high-dimensional vectors via `sentence-transformers/all-MiniLM-L6-v2`.
4.  **Retrieval:** Searches the FAISS index for the top 3 most relevant snippets based on your query.
5.  **Generation:** Passes the snippets and your question to the Groq-hosted LLM for the final synthesis.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9 or higher.
- A Groq Cloud API Key.

### 2. Installation
Clone the repo and install the necessary libraries:
```bash
git clone [https://github.com/YOUR_USERNAME/Thesis-Talker.git](https://github.com/YOUR_USERNAME/Thesis-Talker.git)
cd Thesis-Talker
pip install -r requirements.txt
