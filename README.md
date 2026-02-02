
# **Context-Aware RAG Chatbot**

## **Overview**

This project is a **Context-Aware Chatbot** built using **LangChain** and **Retrieval-Augmented Generation (RAG)**. It can remember conversation history and retrieve information from an external knowledge base or custom corpus, allowing **intelligent and context-aware responses**.

The chatbot is deployed using **Streamlit**, making it easy to interact with through a web interface.

---

## **Features**

* **Contextual memory:** Remembers previous messages in the conversation.
* **RAG-based retrieval:** Fetches answers from a **vectorized document store**.
* **Custom knowledge base:** Supports Wikipedia pages, internal documents, or any custom corpus.
* **LLM integration:** Powered by HuggingFace LLMs.
* **Web deployment:** Streamlit interface for live chat.

---

## **Dataset**

* A **custom corpus** of text documents.
* Can include PDFs, text files, or any knowledge base you want the chatbot to reference.
* Example: `example.txt` contains sample data for testing.

---

## **Installation**

1. **Create a new conda environment (recommended)**

```bash
conda create -n rag_env python=3.11
conda activate rag_env
```

2. **Install required packages**

```bash
pip install --upgrade langchain transformers sentence-transformers torch faiss-cpu streamlit PyPDF2
```

3. **Clone the project (if using GitHub)**

```bash
git clone https://github.com/maroofiums/RAG_Chatbot
cd RAG_Chatbot
```

---

## **Usage**

1. Make sure your **custom corpus** is ready in the project folder (e.g., `example.txt`).
2. Run the chatbot using Streamlit:

```bash
streamlit run main.py
```

3. Open the URL printed in the terminal (usually `http://localhost:8501`) and start chatting.

---

## **Project Structure**

```
RAG_Chatbot/
│
├─ main.py             # Entry point for Streamlit app
├─ rag_setup.py        # RAG setup: vector store, embeddings, LLM
├─ example.txt         # Sample corpus for testing
├─ requirements.txt    # store Python dependencies
└─ README.md           # Project documentation
```

---

## **How it Works**

1. **Embedding Documents:**
   The text corpus is converted into **vector embeddings** using `SentenceTransformer`.

2. **Vector Store:**
   A **FAISS** vector database is used to store embeddings for efficient similarity search.

3. **Context-Aware LLM:**
   Conversation history is stored, allowing the LLM to generate responses considering **previous messages**.

4. **RAG (Retrieval-Augmented Generation):**
   When a user asks a question, the system:

   * Searches the vector store for relevant documents.
   * Combines retrieved content with conversation history.
   * Generates a response using the LLM.

---

## **Skills Gained**

* Building conversational AI with **context memory**
* Document embedding and **vector search**
* Working with **HuggingFace LLMs**
* **RAG pipeline implementation**
* Streamlit-based **web deployment**

---

## **Future Improvements**

* Add support for **multiple document formats** (PDF, DOCX).
* Implement **user authentication** for private document access.
* Deploy chatbot to **cloud (Heroku, AWS, or GCP)** for public access.
* Use **larger LLMs** or **fine-tuned models** for better responses.

---
