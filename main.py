import streamlit as st
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rag_setup import build_vectorstore
from transformers import pipeline

st.set_page_config(page_title="HuggingFace RAG Chatbot", page_icon="🤖")
st.title("📚 HuggingFace Context-Aware RAG Chatbot (CPU)")

DOCS_PATH = "docs"

# Load vectorstore (cached)
@st.cache_resource
def load_vectorstore():
    return build_vectorstore(docs_path=DOCS_PATH)

vectorstore = load_vectorstore()

# Retriever + memory
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# CPU-friendly HF model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-small",  # lightweight, CPU-friendly
    tokenizer="google/flan-t5-small",
    device=-1  # CPU
)

llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

# Chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You: ", "")

if user_input:
    answer = qa_chain.run(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": answer})

# Display chat
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
