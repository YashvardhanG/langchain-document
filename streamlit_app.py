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

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from huggingface_hub import InferenceClient
import os
import tempfile
 
# Load secrets
HUGGINGFACE_TOKEN = st.secrets["huggingface_token"]
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
client = InferenceClient(model=HF_MODEL, token=HUGGINGFACE_TOKEN)
 
# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
# Load initial docs from /docs folder
def load_initial_docs():
    all_docs = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            loader = TextLoader(os.path.join("docs", file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs
 
# Setup vectorstore from documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_docs = text_splitter.split_documents(load_initial_docs())
# vectorstore = FAISS.from_documents(all_docs, embedding_model)
persist_dir = "chroma_db"
vectorstore = Chroma.from_documents(documents = all_docs, embedding = embedding_model, persist_directory = persist_dir)
 
# Function to query HuggingFace Inference API
def query_huggingface(prompt):
    response = client.text_generation(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)
    return response.strip()
 
# UI
st.title("📚 Document QA with Hugging Face")
 
query = st.text_input("Ask a question based on existing documents or upload new one:")
 
if query:
    # Retrieve relevant docs
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
 
    full_prompt = f"""You are an expert AI assistant. Answer the question based on the following context:
                      
                      {context}
 
                      Question: {query}
                      Answer:"""
 
    with st.spinner("Thinking..."):
        answer = query_huggingface(full_prompt)
 
    st.markdown("**Answer:**")
    st.write(answer)
 
# File uploader
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded_file:
    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file_path, "wb") as f:
         f.write(uploaded_file.read())
          
    loader = TextLoader(temp_file_path)
    new_docs = loader.load()
    new_chunks = text_splitter.split_documents(new_docs)
    vectorstore.add_documents(new_chunks)
    vectorstore.persist()
 
    st.success("Document added to knowledge base.")
