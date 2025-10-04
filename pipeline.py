
from langchain_openai import embeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from helper_fncs import (
    OPENAI_API_KEY, 
    GEMINI_API_KEY, 
)

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    from langchain_openai import OpenAIEmbeddings
    print("Creating embeddings...")
    
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
    )

    return vector_store

def ask_and_get_answer(
    vector_store, 
    q,
    k=3
):
    
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GEMINI_API_KEY
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever
    )

    return chain.run(q)
    

def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum(len(enc.encode(text.page_content)) for text in texts)
    return total_tokens, total_tokens / 1000 * 0.0004

