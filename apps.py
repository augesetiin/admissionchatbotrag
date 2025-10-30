# --- IMPORTS --- perfect
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import os
import requests
from bs4 import BeautifulSoup
import time
import re
import string

# --- CONFIGURATION ---
MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "./data/"
ADMISSIONS_URL = "https://siesascn.edu.in/admissions"

# --- SELF-UPDATING KNOWLEDGE BASE ---
@st.cache_data(ttl=86400)
def scrape_and_update_admissions_info():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    output_filename = os.path.join(DATA_PATH, "live_admissions_info.txt")
    if os.path.exists(output_filename):
        if (time.time() - os.path.getmtime(output_filename)) / 3600 < 24:
            return "up-to-date"
    try:
        response = requests.get(ADMISSIONS_URL, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content_div = soup.find('div', class_='entry-content')
        text = content_div.get_text(separator='\n\n', strip=True) if content_div else soup.get_text(separator='\n\n', strip=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(text)
        return "updated"
    except Exception as e:
        st.warning(f"Could not update admissions info: {str(e)}")
        return "error"

# --- CACHED RESOURCES ---
@st.cache_resource
def load_llm_pipeline():
    try:
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        pipe = pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=350, do_sample=True, temperature=0.3, device=device
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

@st.cache_resource
def load_vector_store(_documents):
    try:
        if not _documents:
            st.warning("No documents found in data folder")
            return None
        
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=600, chunk_overlap=100)
        docs = text_splitter.split_documents(_documents)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
        db = FAISS.from_documents(docs, embeddings)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

@st.cache_data
def load_documents():
    try:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            st.warning(f"Created {DATA_PATH} folder. Please add your .txt files there.")
            return []
        
        loader = DirectoryLoader(
            DATA_PATH, glob="*.txt", loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        docs = loader.load()
        if not docs:
            st.warning("No text files found in data folder")
        return docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

# --- QA CHAIN SETUP ---
def setup_qa_chain(_llm, _db):
    if _llm is None or _db is None:
        return None
    
    prompt_template = """You are a helpful and intelligent assistant for SIES College. Your role is to provide clear, accurate, and structured answers based on the provided context.

Instructions:
- Answer only from the context below.
- Write in complete sentences.
- Use headings and bullet points to structure lists.
- If the information is not in the context, politely state that you don't have the information.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = _db.as_retriever(search_kwargs={"k": 3})
    
    # Simple chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | _llm
        | StrOutputParser()
    )
    
    return chain

# --- STREAMLIT APP ---
st.set_page_config(page_title="SIES College Admission Chatbot", page_icon="🎓", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "resources_loaded" not in st.session_state:
    st.session_state.resources_loaded = False

if not st.session_state.resources_loaded:
    with st.spinner("🔄 Loading resources..."):
        scrape_and_update_admissions_info()
        documents = load_documents()
        llm = load_llm_pipeline()
        db = load_vector_store(documents)
        qa_chain = setup_qa_chain(llm, db)
        
        st.session_state.documents = documents
        st.session_state.qa_chain = qa_chain
        st.session_state.resources_loaded = True
        
        if qa_chain is not None:
            st.success("✅ Chatbot ready!")
        else:
            st.warning("⚠️ Chatbot loaded with limited functionality")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎓 SIES Chatbot")
    st.markdown("---")
    st.markdown("### 🚀 Quick Questions")
    example_questions = [
        "Who is the HOD of the IT department?",
        "What documents are required for admission?",
        "What undergraduate courses are offered?",
        "Is there a placement cell?"
    ]
    for question in example_questions:
        if st.button(question, key=question, use_container_width=True):
            st.session_state.user_input = question
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_input = ""
        st.rerun()

# --- MAIN INTERFACE ---
st.title("🎓 SIES College Admission Chatbot")
st.write("Ask questions about courses, faculty, and admission procedures at SIES College.")
st.markdown("---")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def preprocess_input(text):
    text = text.lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))

if st.session_state.resources_loaded:
    if prompt := st.chat_input("Ask about SIES admissions...") or st.session_state.get("user_input"):
        st.session_state.user_input = ""
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                prompt_clean = preprocess_input(prompt)
                response = ""

                if "merit list" in prompt_clean:
                    full_text = "\n".join([doc.page_content for doc in st.session_state.documents])
                    departments = {
                        "B.Com.": ["bcom", "b.com", "commerce"],
                        "B.Sc. IT": ["it", "information technology"],
                        "B.Sc. CS": ["cs", "computer science"],
                        "B.Sc. DS": ["ds", "data science"],
                        "B.Sc. AI": ["ai", "artificial intelligence"],
                        "B.A.M.M.C.": ["bammc", "multimedia", "mass communication"],
                        "B.Com. A&F": ["a&f", "accounts and finance"],
                        "B.Com. B&I": ["b&i", "banking and insurance"],
                        "B.Com. FM": ["fm", "financial markets"],
                        "B.Sc. EVS": ["evs", "environmental science"]
                    }
                    target_dept_name = None
                    for dept_name, keywords in departments.items():
                        if any(keyword in prompt_clean for keyword in keywords):
                            target_dept_name = dept_name
                            break
                    if target_dept_name:
                        escaped_dept = re.escape(target_dept_name)
                        pattern = re.compile(f"Merit List \\d+: {escaped_dept}", re.IGNORECASE)
                        found_lists = pattern.findall(full_text)
                        if found_lists:
                            unique_lists = sorted(list(set(found_lists)))
                            list_str = "\n".join([f"- {item}" for item in unique_lists])
                            response = f"""Here are the merit lists I found for **{target_dept_name}**:

{list_str}

**Total Merit Lists Found:** {len(unique_lists)}

You can find official documents on the [SIES Admissions Portal](https://siesascn.edu.in/admissions)."""
                        else:
                            response = f"I could not find specific merit lists for **{target_dept_name}**. Please check the official portal."
                    else:
                        response = "Please specify a department to check the merit lists."
                
                elif "course" in prompt_clean or "courses" in prompt_clean:
                    full_text = "\n".join([doc.page_content for doc in st.session_state.documents])
                    ug_section = re.search(r"Undergraduate Programs:(.*?)\n\s*\n", full_text, re.DOTALL)
                    pg_section = re.search(r"Postgraduate Programs:(.*?)\n\s*\n", full_text, re.DOTALL)

                    ug_courses = ug_section.group(1).strip().split("\n") if ug_section else []
                    pg_courses = pg_section.group(1).strip().split("\n") if pg_section else []

                    response_parts = []
                    if ug_courses:
                        response_parts.append("### Undergraduate Courses:")
                        for course in ug_courses:
                            response_parts.append(f"- {course.strip()}")
                    if pg_courses:
                        response_parts.append("### Postgraduate Courses:")
                        for course in pg_courses:
                            response_parts.append(f"- {course.strip()}")

                    if not response_parts:
                        response = "I'm sorry, I don't have information about courses."
                    else:
                        response = "\n".join(response_parts)
                
                else:
                    if st.session_state.qa_chain is None:
                        response = "I'm sorry, the chatbot is not properly initialized. Please check your data files."
                    else:
                        try:
                            response = st.session_state.qa_chain.invoke(prompt_clean)
                            
                            if not response or not response.strip() or len(response.strip()) < 10 or any(phrase in response.lower() for phrase in ["i don't have", "no information", "not in the context"]):
                                response = "I'm sorry, but I don't have that information right now."
                        except Exception as e:
                            response = f"I'm sorry, I encountered an error: {str(e)}"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("🔄 Loading... Please wait.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🎓 SIES College Admission Chatbot | Powered by AI"
    "</div>",
    unsafe_allow_html=True
)
