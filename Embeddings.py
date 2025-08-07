print(1)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import openai

print(2)
# Load .env variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
base_url = os.getenv("TOGETHER_BASE_URL")

# ✅ Configure OpenAI client for Together API
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)

print(3)
loader=PyPDFLoader("Al-Kafi.pdf")
pdf_documents=loader.load()


print(4)
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100)
documents = splitter.split_documents(pdf_documents)


print(5)
# 📦 Embedding & Vector Store (use local model to avoid OpenAI embeddings)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents,
    embedding=embeddings,
    persist_directory="chroma_db"  # folder to save the DB
)
retriever = vectorstore.as_retriever()

print(6)
# # 💬 Prompt Template
# prompt = ChatPromptTemplate.from_template(
#     """Answer the question based on the context below:
    
#     Question: {question}

#     Context: {context}
#     """
# )
# print("Embedding and Vector Store creation completed successfully.")