# Import necessary libraries
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveJsonSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import fitz
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import re


# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize global LLM (Large Language Model) with Google Generative AI
global llm
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)

# Function to switch LLM models
def switch_llm(name, model_name):
    global llm
    if name == "gemini":
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.5)
        print(f"Switched LLM to {name} {model_name}")
    if name == "llama":
        llm = ChatGroq(temperature=0, model_name=model_name)
        print(f"Switched LLM to {name} {model_name}")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    # Open the PDF file using fitz (PyMuPDF)
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        # Extract text from each page
        text += page.get_text()
    return text

# Function to load a PDF vector
def load_pdf_vector(file_path):
    global llm
    # Extract text from the PDF file
    text = extract_text_from_pdf(file_path)
    # Remove the PDF file
    os.remove(file_path)
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    # Split the text into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, length_function=len)
    chunks = text_splitter.split_text(text)
    # Create a FAISS vector store from the chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
    # Create a document chain body using the LLM and prompt template
    document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
    # Create a retriever from the vector store
    retriever_body = vector_store.as_retriever()
    # Create a retrieval chain using the retriever and document chain body
    retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
    print("Loaded PDF Vector Store")
    return retrieval_chain_body

# Function to load a vector store from a file
def load_vector_store(name, sourceList):
    global llm
    # Load a FAISS vector store from a file
    vector_store = FAISS.load_local(f"embeddings/{name}", embeddings, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.COSINE)
    # Create a document chain body using the LLM and prompt template
    document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
    # Create a retriever from the vector store with filtering by source
    retriever_body = vector_store.as_retriever(search_kwargs={'filter':{'source':sourceList}})
    # Create a retrieval chain using the retriever and document chain body
    retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
    print("Loaded Vector Store")
    return retrieval_chain_body

# Define a prompt template for chat-based interactions
prompt_template_body = ChatPromptTemplate.from_template("""
Answer the questions in full detail based on the context provided.
your knowledge is limited to the context provided below
Reply with a simple string "no context" if you cannot answer the question.
Context:
{context}
the context is your whole knowledge, you are not to make assumptions and go outside of the context
Remember your job is to use the whole context for answers and do not leave any detail and do not give vague short answers. That is your primary responsibility.
Reply with "no context" if you can not answer the question from the context
.
Question: {input}. Based on the context above answer the question.Use the whole context for answers and answer in full detail.
Just reply with "no context" if the context does not answer the question. You are not supposed to make assumptions and answer questions yourselves
Do not give information outside of context in any way at all as this will have serious problems. Do not make assumptions either just answer from context
If the question can not be answered by context and you cant answer any question then just reply "no context".
Do not try to explain anything when the question cannot be answered from the context. just return a single string "no context"
""")

# Function to extract URLs from a text
def extract_urls(text):
    pattern = r'https?://[^,\s\n\]]*'
    urls = re.findall(pattern, text)
    unique_urls = list(set(urls))
    return unique_urls