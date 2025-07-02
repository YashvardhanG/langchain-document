import PyPDF2
import os
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from nltk.tokenize import sent_tokenize
from langchain.vectorstores import FAISS  
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nltk
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
    embeddings = HuggingFaceEmbeddings()
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
    st.header("Document AI")

    st.write("Dataset Loaded")
    base_db = load_base_knowledge()

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
        llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-large", task="text2text-generation", model_kwargs={"temperature": 0.7, "max_length": 512, "top_p":0.95, "repetition_penalty":1.1})
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.markdown("#### 📖 Answer:")
        st.write(response)


if __name__ == '__main__':
    main()
