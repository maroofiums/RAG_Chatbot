from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def build_vectorstore(docs_path="docs"):
    # Load TXT and PDF
    txt_loader = DirectoryLoader(docs_path, glob="*.txt")
    pdf_loader = DirectoryLoader(docs_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = txt_loader.load() + pdf_loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # HF embeddings (SBERT)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
