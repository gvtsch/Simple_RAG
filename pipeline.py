
from langchain_openai import embeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from helper_fncs import (
    OPENAI_API_KEY,
    GEMINI_API_KEY, 
)

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=GEMINI_API_KEY
    )
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )

    # Create a prompt template for the QA task
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a simple RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run the chain and return the answer
    response = rag_chain.invoke(q)
    return response
    

def calculate_embedding_cost(texts):
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum(len(enc.encode(text.page_content)) for text in texts)
    return total_tokens, total_tokens / 1000 * 0.0004

