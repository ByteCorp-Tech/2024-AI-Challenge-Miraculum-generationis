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



notion_corpus = load_corpus('corpus/notion_corpus.json')
notion_cleaned = remove_keys_from_dict(notion_corpus, keys_to_remove)
jira_corpus = load_corpus('corpus/jira_corpus.json')
github_corpus = load_corpus('corpus/github_corpus.json')
website_text=custom_chunking_website('website_data')


notion_text = 'Notion:\n'.join([parse_dict(page) for page in notion_cleaned])
jira_text = flatten_corpus(jira_corpus)
github_text = flatten_repo_data(github_corpus)
all_text='\n'.join([github_text, notion_text, jira_text,website_text])




def save_embeddings(text,name):
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=600, length_function=len)
    chunks = text_splitter.split_text(text)
    vector_store = Chroma.from_texts(chunks, embeddings,persist_directory=f"embeddings/{name}")



save_embeddings(all_text,'all')



