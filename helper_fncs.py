import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if OPENAI_API_KEY:
    print("OpenAI API Key loaded successfully.")
else: 
    print("Failed to load API Key.")

if GEMINI_API_KEY:
    print("Gemini API Key loaded successfully.")
else:
    print("Failed to load Gemini API Key.")




def load_document(file):
    import os
    name, ext = os.path.splitext(file)

    if ext == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading document {file}...")
        loader = PyPDFLoader(file)
    elif ext == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading document {file}...")
        loader = Docx2txtLoader(file)
    elif ext == '.txt':
        from langchain.document_loaders import TextLoader
        print(f"Loading document {file}...")
        loader = TextLoader(file, encoding='utf-8')
    else:
        print(f"Unsupported file format: {ext}")
        return None
    
    data = loader.load()
    return data

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    st.success("History cleared!")