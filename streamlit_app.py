# import streamlit as st
# from openai import OpenAI

# # Show title and description.
# st.title("📄 Document question answering")
# st.write(
#     "Upload a document below and ask a question about it – GPT will answer! "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
# )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

#     # Create an OpenAI client.
#     client = OpenAI(api_key=openai_api_key)

#     # Let the user upload a file via `st.file_uploader`.
#     uploaded_file = st.file_uploader(
#         "Upload a document (.txt or .md)", type=("txt", "md")
#     )

#     # Ask the user for a question via `st.text_area`.
#     question = st.text_area(
#         "Now ask a question about the document!",
#         placeholder="Can you give me a short summary?",
#         disabled=not uploaded_file,
#     )

#     if uploaded_file and question:

#         # Process the uploaded file and question.
#         document = uploaded_file.read().decode()
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Here's a document: {document} \n\n---\n\n {question}",
#             }
#         ]

#         # Generate an answer using the OpenAI API.
#         stream = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             stream=True,
#         )

#         # Stream the response to the app using `st.write_stream`.
#         st.write_stream(stream)

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import streamlit as st
# # from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
# import chromadb
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain.document_loaders import TextLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from huggingface_hub import InferenceClient
# import os
# import tempfile
 
# # Load secrets
# HUGGINGFACE_TOKEN = st.secrets["huggingface_token"]
# # HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
# HF_MODEL = "google/flan-t5-base"
# client = InferenceClient(model=HF_MODEL, token=HUGGINGFACE_TOKEN)
 
# # Load embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
# # Load initial docs from /docs folder
# def load_initial_docs():
#     all_docs = []
#     for file in os.listdir("docs"):
#         if file.endswith(".pdf"):
#             # loader = TextLoader(os.path.join("docs", file))
#             loader = PyPDFLoader(os.path.join("docs", file))
#             docs = loader.load()
#             all_docs.extend(docs)
#     return all_docs
 
# # Setup vectorstore from documents
# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# all_docs = text_splitter.split_documents(load_initial_docs())
# # vectorstore = FAISS.from_documents(all_docs, embedding_model)
# persist_dir = "chroma_db"
# vectorstore = Chroma.from_documents(documents = all_docs, embedding = embedding_model, persist_directory = persist_dir)
 
# # Function to query HuggingFace Inference API
# def query_huggingface(prompt):
#     response = client.text_generation(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)
#     return response.strip()
 
# # UI
# st.title("📚 Document QA with Hugging Face")
 
# query = st.text_input("Ask a question based on existing documents or upload new one:")
 
# if query:
#     # Retrieve relevant docs
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#     docs = retriever.get_relevant_documents(query)
#     context = "\n\n".join([doc.page_content for doc in docs])
 
#     full_prompt = f"""You are an expert AI assistant. Answer the question based on the following context:
                      
#                       {context}
 
#                       Question: {query}
#                       Answer:"""
 
#     with st.spinner("Thinking..."):
#         answer = query_huggingface(full_prompt)
 
#     st.markdown("**Answer:**")
#     st.write(answer)
 
# # File uploader
# uploaded_file = st.file_uploader("Upload a file")
# if uploaded_file is not None:
#     temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
#     with open(temp_file_path, "wb") as f:
#          f.write(uploaded_file.read())
          
#     # loader = TextLoader(temp_file_path)
#     loader = PyPDFLoader(temp_file_path)
#     new_docs = loader.load()
#     new_chunks = text_splitter.split_documents(new_docs)
#     vectorstore.add_documents(new_chunks)
#     vectorstore.persist()
 
#     st.success("Document added to knowledge base.")

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import PyPDF2
from io import BytesIO
import streamlit as st
from nltk.tokenize import sent_tokenize
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
# import nltk

# nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
# nltk.data.path.append(nltk_data_dir)
 
# try:
#     nltk.data.find("tokenizers/punkt")
# except LookupError:
#     nltk.download("punkt", download_dir=nltk_data_dir)
#     nltk.download("punkt")
import nltk
nltk.download('punkt_tab')

#import re
#def sent_tokenize(text):
#    return re.split(r'(?<=[.!?])\s+', text.strip())

# Load Hugging Face API token from secrets
HUGGINGFACE_TOKEN = "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"
#HF_MODEL = "google/flan-t5-large"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
client = InferenceClient(model=HF_MODEL, token=HUGGINGFACE_TOKEN)
 
# VectorDB directory
persist_dir = "chroma_db"
 
# Extract text from PDFs
def extract_text_from_pdf(pdf_bytes):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text
 
# Process uploaded PDFs into text
def process_uploaded_pdfs(docs):
    all_text = ""
    for doc in docs:
        if doc.type == "application/pdf":
            doc_text = extract_text_from_pdf(doc.read())
            sentences = sent_tokenize(doc_text)
            all_text += "\n".join(sentences) + "\n"
    return all_text
 
# Split and embed documents
def get_vectorstore_from_text(text):
    if not text:
        return None
    sentences = sent_tokenize(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents(sentences)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb
 
# Load preloaded docs
def load_base_knowledge():
    base_folder = "docs"
    all_text = ""
    for file_name in os.listdir(base_folder):
        if file_name.endswith(".pdf"):
            with open(os.path.join(base_folder, file_name), "rb") as f:
                text = extract_text_from_pdf(f.read())
                all_text += "\n".join(sent_tokenize(text)) + "\n"
    return get_vectorstore_from_text(all_text)
 
# Query HuggingFace Inference API
def query_huggingface(prompt):
    response = client.text_generation(prompt, max_new_tokens=512, temperature=0.7, top_p=0.95, repetition_penalty=1.1)
    return response.strip()
 
# Streamlit UI
def main():
    st.set_page_config(page_title="Document QA with Hugging Face", page_icon="🧠")
    st.title("📄 Document Question Answering (Hugging Face API)")
 
    with st.spinner("Loading base documents..."):
        vectorstore = load_base_knowledge()
 
    uploaded_docs = st.file_uploader("📁 Upload additional PDFs", type=["pdf"], accept_multiple_files=True)
 
    if uploaded_docs:
        with st.spinner("Processing uploaded documents..."):
            uploaded_text = process_uploaded_pdfs(uploaded_docs)
            uploaded_db = get_vectorstore_from_text(uploaded_text)
            if uploaded_db:
                #vectorstore.merge_from(uploaded_db)
                new_docs = uploaded_db.similarity_search("")
                vectorstore.add_documents(new_docs)
                vectorstore.persist()
 
    user_question = st.text_input("Ask a question:")
    if user_question and vectorstore:
        docs = vectorstore.similarity_search(user_question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer the following question based on the provided context.
 
                     Context:
                     {context}
                      
                     Question: {user_question}
                     Answer:"""
 
        with st.spinner("Generating answer..."):
            answer = query_huggingface(prompt)
            st.markdown("### 🧠 Answer:")
            st.write(answer)
 
if __name__ == "__main__":
    main()


