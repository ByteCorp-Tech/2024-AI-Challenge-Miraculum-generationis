import streamlit as st
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import langchain
from langchain_openai import ChatOpenAI
FAISS.allow_dangerous_deserialization = True
import fitz
import os
from prompts import prompt_template_body, prompt_template_url
import base64

st.set_page_config(page_title="AI Chat Assistant", layout="wide", page_icon=":robot:", initial_sidebar_state="expanded")
load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# llm = ChatOpenAI(model="gpt-4", temperature=0)

st.title("AI Assistant")
st.markdown("### Ask me anything!")

with st.sidebar:
    st.header("Settings")
    st.session_state.upload_file = st.checkbox("Upload File")
    st.session_state.toggle_notion = st.checkbox("Notion", value=True)
    st.session_state.toggle_jira = st.checkbox("JIRA", value=True)
    st.session_state.toggle_github = st.checkbox("GitHub", value=True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if st.session_state.upload_file:
    uploaded_file = st.file_uploader("Upload your file", type=["pdf"])
    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        text = extract_text_from_pdf(file_path)
        os.remove(file_path)
        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, embeddings)
else:
    if st.session_state.toggle_notion and st.session_state.toggle_jira and st.session_state.toggle_github:
        vector_store = FAISS.load_local("embeddings/github_notion_jira", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_jira and st.session_state.toggle_github:
        vector_store = FAISS.load_local("embeddings/jira_github", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_notion and st.session_state.toggle_github:
        vector_store = FAISS.load_local("embeddings/notion_github", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_notion and st.session_state.toggle_jira:
        vector_store = FAISS.load_local("embeddings/notion_jira", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_github:
        vector_store = FAISS.load_local("embeddings/github", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_jira:
        vector_store = FAISS.load_local("embeddings/jira", embeddings, allow_dangerous_deserialization=True)
    elif st.session_state.toggle_notion:
        vector_store = FAISS.load_local("embeddings/notion", embeddings, allow_dangerous_deserialization=True)

try:
    document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
    retriever_body = vector_store.as_retriever()
    retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)

    document_chain_url = create_stuff_documents_chain(llm, prompt_template_url)
    retriever_url = vector_store.as_retriever()
    retrieval_chain_url = create_retrieval_chain(retriever_url, document_chain_url)
except:
    print("Waiting for file upload")

def query_assistant_body(input):
    response = retrieval_chain_body.invoke({"input": input})
    context=response.get("context","No context available")
    print(context)
    return response["answer"]

def query_assistant_url(input):
    response = retrieval_chain_url.invoke({"input": input})
    return response["answer"]

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question:", "", key="user_input")
if st.button("Send", key="send_button"):
    if user_input:
        answer = query_assistant_body(user_input)
        urls = query_assistant_url(user_input)
        urls = json.loads(urls)
        links = []
        for url in urls:
            links.append(f"[{url}]({url})\n")
        if links:
            st.session_state.history.append({"user": user_input, "assistant": answer + "\n### Related Links\n" + "\n".join(links)})
        else:
            st.session_state.history.append({"user": user_input, "assistant": answer})
        st.session_state.history.reverse()
        chat_area = st.empty()
        for chat in st.session_state.history:
            chat_area.markdown(f"**User:** {chat['user']}\n{chat['assistant']}\n")
        st.session_state.history.reverse()

