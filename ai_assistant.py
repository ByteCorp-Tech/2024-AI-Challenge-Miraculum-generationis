import streamlit as st
import json
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from github_helper_functions import flatten_repo_data
from jira_helper_functions import flatten_corpus
from notion_helper_functions import parse_dict, remove_keys_from_dict,keys_to_remove
import langchain
from langchain_openai import ChatOpenAI
FAISS.allow_dangerous_deserialization = True
import tempfile
import fitz
import os
import re
load_dotenv()





embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# llm = ChatOpenAI(model="gpt-4", temperature=0)


st.set_page_config(layout="wide")
st.title("AI Assistant")
def on_change():
    if st.session_state.toggle_notion or st.session_state.toggle_jira or st.session_state.toggle_github:
        st.session_state.upload_file = False
    st.session_state["checked_state"] = (st.session_state.toggle_notion, st.session_state.toggle_jira, st.session_state.toggle_github)



def on_file_upload_change():
    if st.session_state.upload_file:
        st.session_state.toggle_notion = False
        st.session_state.toggle_jira = False
        st.session_state.toggle_github = False
    st.session_state["checked_state"] = (st.session_state.toggle_notion, st.session_state.toggle_jira, st.session_state.toggle_github)



if "upload_file" not in st.session_state:
    st.session_state.upload_file = False

if "checked_state" not in st.session_state:
    st.session_state["checked_state"] = (True, True, True)

if "toggle_notion" not in st.session_state:
    st.session_state["toggle_notion"] = True
if "toggle_jira" not in st.session_state:
    st.session_state["toggle_jira"] = True
if "toggle_github" not in st.session_state:
    st.session_state["toggle_github"] = True

st.sidebar.checkbox("Upload File", value=st.session_state.upload_file, on_change=on_file_upload_change, key="upload_file")
st.sidebar.checkbox("Notion", value=st.session_state.toggle_notion, on_change=on_change, key="toggle_notion", disabled=st.session_state.upload_file)
st.sidebar.checkbox("JIRA", value=st.session_state.toggle_jira, on_change=on_change, key="toggle_jira", disabled=st.session_state.upload_file)
st.sidebar.checkbox("GitHub", value=st.session_state.toggle_github, on_change=on_change, key="toggle_github", disabled=st.session_state.upload_file)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text



prompt_template_body = ChatPromptTemplate.from_template("""Answer the questions in full detail based on the context provided. Reply with a simple string "no context" if you cannot answer the question.
Context:
{context}

Remember your job is to use the whole context for answers and do not leave any detail and do not give vague short answers. That is your primary responsibility.
                                                                                                     Reply with "no context" if you can not answer the question from the context                                                               

                                                        . 
Question: {input}. Based on the context above answer the question.Use the whole context for answers and answer in full detail.

                    If the question can not be answered by context and you cant answer any question then just reply "no context".
""")




if st.session_state.upload_file:
    uploaded_file = st.file_uploader("Upload your file", type=["pdf"])
    print("PDF Mode")
    if uploaded_file is not None:
        file_path = uploaded_file.name
        print(file_path)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        text = extract_text_from_pdf(file_path)
        os.remove(file_path)
        embeddings = OpenAIEmbeddings()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, embeddings)
    try:
        document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
        retriever_body = vector_store.as_retriever()
        retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)

    except:
        print("Waiting for file upload")
        
elif st.session_state["checked_state"] == (True, True, True):
    print("all 3")
    vector_store = FAISS.load_local("embeddings/github_notion_jira", embeddings, allow_dangerous_deserialization=True)
    try:
        document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
        retriever_body = vector_store.as_retriever()
        retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)


    except:
        print("Waiting for file upload")

elif st.session_state["checked_state"][1] and st.session_state["checked_state"][2]:
    print("jira and github")
    vector_store = FAISS.load_local("embeddings/jira_github", embeddings, allow_dangerous_deserialization=True)

elif st.session_state["checked_state"][0] and st.session_state["checked_state"][2]:
    print("notion and github")
    vector_store = FAISS.load_local("embeddings/notion_github", embeddings, allow_dangerous_deserialization=True)

elif st.session_state["checked_state"][0] and st.session_state["checked_state"][1]:
    print("notion and jira")
    vector_store = FAISS.load_local("embeddings/notion_jira", embeddings, allow_dangerous_deserialization=True)

elif st.session_state["checked_state"][2]:
    print("only github")
    vector_store = FAISS.load_local("embeddings/github", embeddings, allow_dangerous_deserialization=True)

elif st.session_state["checked_state"][1]:
    print("only jira")
    vector_store = FAISS.load_local("embeddings/jira", embeddings, allow_dangerous_deserialization=True)

elif st.session_state["checked_state"][0]:
    print("only notion")
    vector_store = FAISS.load_local("embeddings/notion", embeddings, allow_dangerous_deserialization=True)
    try:
        document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
        retriever_body = vector_store.as_retriever()
        retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
    except:
        print("Waiting for file upload")




def extract_urls(text):
    pattern = r'https?://[^,\s\n\]]*'
    urls = re.findall(pattern, text)
    unique_urls = list(set(urls))
    return unique_urls

def query_assistant_body(input):
    response = retrieval_chain_body.invoke({"input": input})
    context=response.get("context","No context available")
    print(context)
    context_text=""
    for document in context:
        context_text+=document.page_content
    urls=extract_urls(context_text)
    print(urls)
    return response["answer"],urls


user_input = st.text_input("Ask your question:", "")
if st.button("Send"):
    if user_input:
        answer,urls = query_assistant_body(user_input)
        links = []
        for url in urls:
            links.append(f"[{url}]({url})\n")
        if links:
            st.markdown("### Related Links")
            st.markdown("\n".join(links))
        else:
            st.markdown("### Related Links")
            st.markdown("No relevant links found.")
        if answer == "no context" and len(urls)!=0:
            answer="I have only found the relevant links but cannot answer the question from context available. Please check the links provided"
        if answer=="no context" and len(urls)==0:
            answer="I have not found any relevant links or any answer from context"
        else:
            answer=answer.replace(",","\n")
        st.text_area("Answer:", value=answer, height=300, key="bot_response")
        
        