import streamlit as st
import json
from dotenv import load_dotenv
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter,RecursiveJsonSplitter,RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from github_helper_functions import flatten_repo_data
from jira_helper_functions import flatten_corpus
from langchain_chroma import Chroma
from notion_helper_functions import parse_dict, remove_keys_from_dict,keys_to_remove
from website_helper_functions import custom_chunking_website
from langchain_openai.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import langchain
FAISS.allow_dangerous_deserialization = True

load_dotenv()


def load_corpus(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def split_chunks(strings, chunk_size=1500, overlap=300):
    updated_list = []
    for s in strings:
        while len(s) > chunk_size:
            updated_list.append(s[:chunk_size])
            s = s[chunk_size - overlap:]
        if s:
            updated_list.append(s)
    
    return updated_list





notion_corpus = load_corpus('corpus/notion_corpus.json')
notion_cleaned = remove_keys_from_dict(notion_corpus, keys_to_remove)
jira_corpus = load_corpus('corpus/jira_corpus.json')
github_corpus = load_corpus('corpus/github_corpus.json')



notion_text = ['notion\n' + parse_dict(page) for page in notion_cleaned]
website_text=custom_chunking_website('website_data')
jira_text = flatten_corpus(jira_corpus)
github_text = flatten_repo_data(github_corpus)
notion_text=split_chunks(notion_text,1500,300)
# all_text='\n'.join([github_text,jira_text,website_text])





def save_embeddings(chunks,name):
    embeddings = OpenAIEmbeddings()
    print(chunks[0])
    print("\n\n\n\n")
    print(chunks[1])
    vector_store = Chroma.from_texts(chunks, embeddings,persist_directory=f"embeddings/{name}")

save_embeddings(notion_text,'Notion')
save_embeddings(jira_text,'Jira')
save_embeddings(github_text,'Github')
save_embeddings(website_text,'Website')


