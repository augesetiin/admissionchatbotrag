# --- best IMPORTS ---
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
# NEW IMPORTS for web scraping
import requests
from bs4 import BeautifulSoup
import time

# --- CONFIGURATION ---
MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "./data/"
ADMISSIONS_URL = "https://siesascn.edu.in/admissions"

# --- OPTIMIZED: SELF-UPDATING KNOWLEDGE BASE ---
@st.cache_data(ttl=86400)  # Cache for 24 hours
def scrape_and_update_admissions_info():
    """
    Scrapes the admissions page for the latest info and saves it to a text file.
    Cached to avoid repeated scraping.
    """
    output_filename = os.path.join(DATA_PATH, "live_admissions_info.txt")
    
    # Check if the file exists and is recent (less than 24 hours old)
    if os.path.exists(output_filename):
        file_mod_time = os.path.getmtime(output_filename)
        if (time.time() - file_mod_time) / 3600 < 24:
            return "up-to-date"
    
    try:
        response = requests.get(ADMISSIONS_URL, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content area of the page
        content_div = soup.find('div', class_='entry-content')
        
        if content_div:
            text = content_div.get_text(separator='\n\n', strip=True)
        else:
            text = soup.get_text(separator='\n\n', strip=True)
        
        # Limit text length to prevent oversized files
        text = text[:5000] if len(text) > 5000 else text
            
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        
        return "updated"
        
    except Exception as e:
        return f"error: {e}"

# --- OPTIMIZED: CACHED RESOURCES ---
@st.cache_resource
def load_llm_pipeline():
    """Load and cache the language model pipeline"""
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Optimized pipeline with better parameters
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256,  # Reduced for faster response
        do_sample=True,
        temperature=0.3,     # More focused responses
        device=device
    )
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def load_vector_store():
    """Load and cache the vector store"""
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="*.txt", 
        loader_cls=TextLoader, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()
    
    # OPTIMIZED: Smaller chunks for faster processing
    text_splitter = CharacterTextSplitter(
        separator="\n\n", 
        chunk_size=400,      # Reduced from 1000
        chunk_overlap=50     # Reduced from 100
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.from_documents(docs, embeddings)
    return db

# --- OPTIMIZED: QA CHAIN SETUP ---
def setup_qa_chain(_llm, _db):
    """Setup the QA chain with optimized prompt"""
    # OPTIMIZED: More concise prompt template
    prompt_template = """You are a SIES College admission assistant. Answer based on the context below.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # OPTIMIZED: Return fewer documents for faster processing
    retriever = _db.as_retriever(search_kwargs={"k": 2})  # Reduced from 3
    
    return RetrievalQA.from_chain_type(
        llm=_llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT}
    )

# --- OPTIMIZED: CONTEXT TRUNCATION ---
def truncate_context(context, max_tokens=400):
    """Truncate context to prevent token limit issues"""
    words = context.split()
    if len(words) > max_tokens:
        return " ".join(words[:max_tokens]) + "..."
    return context

# --- STREAMLIT APP ---
st.set_page_config(
    page_title="SIES College Admission Chatbot", 
    page_icon="🎓", 
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resources_loaded" not in st.session_state:
    st.session_state.resources_loaded = False

# --- OPTIMIZED: Load resources only once ---
if not st.session_state.resources_loaded:
    with st.spinner("🔄 Loading chatbot resources..."):
        # Update admissions info
        scrape_status = scrape_and_update_admissions_info()
        
        # Load models
        llm = load_llm_pipeline()
        db = load_vector_store()
        qa_chain = setup_qa_chain(llm, db)
        
        # Store in session state
        st.session_state.llm = llm
        st.session_state.db = db
        st.session_state.qa_chain = qa_chain
        st.session_state.resources_loaded = True
        st.success("✅ Chatbot loaded successfully!")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎓 SIES Chatbot")
    st.markdown("---")
    st.markdown("### 🚀 Quick Questions")
    
    example_questions = [
        "What are the latest admission notices?",
        "Who is the Head of the Commerce department?",
        "What documents are required for admission?",
        "What are the admission deadlines?",
        "What courses are offered?"
    ]
    
    for question in example_questions:
        if st.button(question, key=question, use_container_width=True):
            st.session_state.user_input = question
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.rerun()
    
    # Performance info
    st.markdown("---")
    st.markdown("### ⚡ Performance")
    st.markdown("- **Model**: FLAN-T5 Base")
    st.markdown("- **Status**: Optimized for Speed")

# --- MAIN INTERFACE ---
st.title("🎓 SIES College Admission Chatbot")
st.write("Your fast AI assistant for SIES College (Nerul) admissions.")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if st.session_state.resources_loaded:
    if prompt := st.chat_input("Ask about admissions...") or st.session_state.get("user_input"):
        # Clear user input
        if "user_input" in st.session_state:
            st.session_state.user_input = ""
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # OPTIMIZED: Process query with truncation
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    response = result["result"]
                    
                    # Clean up response
                    if response.startswith("Answer:"):
                        response = response[7:].strip()
                    
                except Exception as e:
                    response = f"I apologize, but I encountered an error: {str(e)[:100]}..."
                
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # OPTIMIZED: Show sources only if available and not too many
        if 'result' in locals() and result and "source_documents" in result:
            if len(result["source_documents"]) <= 3:  # Limit sources display
                with st.expander("🔍 View Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.info(f"**Source {i+1}**: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                        # Truncate source content for better performance
                        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        st.write(content)
else:
    st.warning("🔄 Chatbot is loading. Please wait...")
    st.info("This may take a few moments on first run.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎓 SIES College Admission Chatbot | Powered by AI"
    "</div>", 
    unsafe_allow_html=True
)
