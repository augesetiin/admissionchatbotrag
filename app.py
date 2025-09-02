# --- Imports ---
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Use the new, recommended Hugging Face integrations
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

# --- Model Name ---
# ✅ Changed to a smaller model to prevent freezing on systems with limited RAM
MODEL_NAME = "google/flan-t5-small"

# --- 1. Load LLM ---
@st.cache_resource
def load_llm_pipeline():
    print(f"🔄 Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Use the 'text2text-generation' pipeline for Flan-T5
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Use the new HuggingFacePipeline wrapper
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ LLM pipeline loaded successfully!")
    return llm

# --- 2. Load Documents and Create Vector Store ---
@st.cache_resource
def load_vector_store():
    print("📂 Loading documents from './data'...")
    # Use DirectoryLoader for both PDF and TXT files
    pdf_loader = DirectoryLoader('./data/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader('./data/', glob="**/*.txt", loader_cls=TextLoader)
    
    documents = pdf_loader.load() + txt_loader.load()
    print(f"📄 Found {len(documents)} document(s)")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(docs)} chunks")

    # Create embeddings using the new HuggingFaceEmbeddings
    print("🔄 Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    print("✅ Vector store created!")
    return db

# --- Streamlit User Interface ---
st.title("📄 Chat with Your Documents")
st.write("This chatbot uses your local text and PDF files to answer questions.")

# Load everything
llm = load_llm_pipeline()
db = load_vector_store()

# Create the QA chain
retriever = db.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
print("✅ QA Chain ready!")

# Get user input
user_question = st.text_input("Ask a question about your documents:")

if user_question:
    with st.spinner("Finding an answer..."):
        # Use the modern .invoke() method
        result = qa_chain.invoke({"query": user_question})
        
        # Access the answer from the result dictionary
        st.write("### Answer")
        st.write(result["result"])
