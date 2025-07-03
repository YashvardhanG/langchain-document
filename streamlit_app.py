import os
import streamlit as st
import PyPDF2
import nltk
import time
from io import BytesIO
from dotenv import load_dotenv
# from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFacePipeline, HuggingFaceHub
from nltk.tokenize import sent_tokenize
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
nltk.download('punkt_tab')

def extract_text_from_pdf(pdf_bytes):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text


def process_uploaded_pdfs(docs):
    all_text = ""
    for doc in docs:
        if doc.type == "application/pdf":
            doc_text = extract_text_from_pdf(doc.read())
            sentences = sent_tokenize(doc_text)
            all_text += "\n".join(sentences) + "\n"
    return all_text


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not chunks:
        return None
    return FAISS.from_documents(chunks, embeddings)


def load_base_knowledge():
    base_folder = "docs"
    all_text = ""
    for file_name in os.listdir(base_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, "rb") as f:
                text = extract_text_from_pdf(f.read())
                all_text += "\n".join(sent_tokenize(text)) + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents(sent_tokenize(all_text))
    return get_vectorstore(chunks)


def main():
    load_dotenv()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["huggingface_token"]
    st.set_page_config(page_title="Document AI", page_icon="⚙️")

    col1, col2 = st.columns([1,5])
    with col1:
        st.image("BH_Logo_Horizontal_White.png", width = 250)
    with col2:
        st.header("Document AI")

    status_holder = st.empty()
    status_holder.write("⌛ Loading Dataset")
    base_db = load_base_knowledge()
    status_holder.write("✅ Dataset Loaded")
    time.sleep(5)
    status_holder.empty()

    uploaded_pdfs = st.file_uploader("📁 Upload Document (optional)", type=["pdf"], accept_multiple_files=True)
    
    final_db = base_db

    if uploaded_pdfs:
        uploaded_text = process_uploaded_pdfs(uploaded_pdfs)
        sentences = sent_tokenize(uploaded_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        new_chunks = text_splitter.create_documents(sentences)
        new_db = get_vectorstore(new_chunks)

        if new_db:
            base_db.merge_from(new_db)
            final_db = base_db

    user_question = st.text_input("Enter your prompt here:")

    if user_question and final_db:
        docs = final_db.similarity_search(user_question)
        llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"max_length": 512})
        # llm = HuggingFacePipeline.from_model_id(model_id="HuggingFaceH4/zephyr-7b-beta", task="text2text-generation", model_kwargs={"temperature": 0.7, "max_new_tokens": 256, "top_p": 0.95})
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.markdown("#### 📖 Answer:")
        st.write(response)


if __name__ == '__main__':
    main()
