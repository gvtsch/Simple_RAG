import keyring
import os
import shutil
import streamlit as st
from pipeline import (
    chunk_data, 
    create_embeddings, 
    ask_and_get_answer, 
    calculate_embedding_cost,
)

from helper_fncs import (
    load_document, 
    clear_history
)



def main():
    st.set_page_config(page_title="LLM QA Chatbot", layout="wide")
    st.subheader("LLM Question Answering App 🤖")


    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    

    with st.sidebar:
        st.header("1. Data Processing")
        uploaded_file = st.file_uploader(
            "Upload a document", type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=clear_history 
        )
        
        k = st.number_input(
            "k (Top Chunks)", 
            min_value=1, 
            max_value=20, 
            value=3, 
            on_change=clear_history
        )
        
        add_data = st.button(
            'Add Data', 
            on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding data...'):
                data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open (file_name, 'wb') as f:
                    f.write(data)

                data = load_document(file_name)
                try:
                    os.remove(file_name) 
                except Exception:
                    pass

                if data:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.success(f'Chunks created: {len(chunks)}.')
                    
                    tokens, embeddings_cost = calculate_embedding_cost(chunks)
                    st.info(f'Token costs: {tokens}, Estimated costs: ${embeddings_cost:.4f}')

                    vector_store = create_embeddings(chunks)
                    st.session_state['vs'] = vector_store
                    st.success("Data successfully prepared for querying!")

                    st.session_state['messages'].append({
                        "role": "assistant", 
                        "content": f"Your documents ({uploaded_file.name}) have been processed. You can now ask questions."
                    })
                    st.rerun()

    st.header("Your Chat")

    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know or do?"):
        
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if 'vs' not in st.session_state:
            with st.chat_message("assistant"):
                st.warning("Please upload documents first and click 'Add Data'.")
            st.session_state['messages'].append({"role": "assistant", "content": "Please upload documents first and click 'Add Data'."})
            return

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                vector_store = st.session_state['vs']
                current_k = st.session_state.get('k', 3)                
                try:
                    answer = ask_and_get_answer(vector_store, prompt, k=current_k)
                    st.markdown(answer)
                    
                    st.session_state['messages'].append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {e}"
                    st.error(error_msg)
                    st.session_state['messages'].append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()