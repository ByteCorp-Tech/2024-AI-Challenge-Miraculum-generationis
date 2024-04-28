import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter,RecursiveJsonSplitter,RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import ChatOpenAI
import fitz
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import re
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# llm = ChatOpenAI(model="gpt-4", temperature=0)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_pdf_vector(file_path):
        text = extract_text_from_pdf(file_path)
        os.remove(file_path)
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, length_function=len)
        chunks = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, embeddings)
        document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
        retriever_body = vector_store.as_retriever()
        retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
        return retrieval_chain_body



def load_vector_store(name,sourceList):
    vector_store = FAISS.load_local(f"embeddings/{name}", embeddings,allow_dangerous_deserialization = True,distance_strategy=DistanceStrategy.COSINE)
    # with open("context.txt","w",encoding="utf-8") as f: 
    #     results = vector_store.similarity_search("Who is Kaamla Salman?", filter=dict(source="website"))
    #     for doc in results:
    #         print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
    #         f.write(doc.page_content)
    document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
    retriever_body = vector_store.as_retriever(search_kwargs={'filter':{'source':sourceList}})
    retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
    return retrieval_chain_body



prompt_template_body = ChatPromptTemplate.from_template("""Answer the questions in full detail based on the context provided. Reply with a simple string "no context" if you cannot answer the question.
Context:
{context}

Remember your job is to use the whole context for answers and do not leave any detail and do not give vague short answers. That is your primary responsibility.
                                                                                                     Reply with "no context" if you can not answer the question from the context                                                               

                                                        . 
Question: {input}. Based on the context above answer the question.Use the whole context for answers and answer in full detail.

                    If the question can not be answered by context and you cant answer any question then just reply "no context".
""")

        
def extract_urls(text):
    pattern = r'https?://[^,\s\n\]]*'
    urls = re.findall(pattern, text)
    unique_urls = list(set(urls))
    return unique_urls





        