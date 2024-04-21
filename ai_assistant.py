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

load_dotenv()





embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# llm = ChatOpenAI(model="gpt-4", temperature=0)

# Streamlit interface setup
st.set_page_config(layout="wide")
st.title("AI Assistant")
def on_change():
    st.session_state["checked_state"] = (st.session_state.toggle_notion, st.session_state.toggle_jira, st.session_state.toggle_github)

# Initialize session state variables if they don't exist
if "checked_state" not in st.session_state:
    st.session_state["checked_state"] = (True, True, True)

if "toggle_notion" not in st.session_state:
    st.session_state["toggle_notion"] = True
if "toggle_jira" not in st.session_state:
    st.session_state["toggle_jira"] = True
if "toggle_github" not in st.session_state:
    st.session_state["toggle_github"] = True

# Define the checkboxes
st.sidebar.checkbox("Notion", value=st.session_state.toggle_notion, on_change=on_change, key="toggle_notion")
st.sidebar.checkbox("JIRA", value=st.session_state.toggle_jira, on_change=on_change, key="toggle_jira")
st.sidebar.checkbox("GitHub", value=st.session_state.toggle_github, on_change=on_change, key="toggle_github")

if st.session_state["checked_state"] == (True, True, True):
    print("all 3")
    vector_store = FAISS.load_local("embeddings/github_notion_jira", embeddings, allow_dangerous_deserialization=True)

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


prompt_template_body = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. 
Context:
{context}

Based on the context above answer the question.,
Question: {input}.
""")


prompt_template_url = ChatPromptTemplate.from_template("""You are a url extractor. Whenever a user asks a question your job not to answer the question but
                                                       to extract the urls for the intended answer. Extract urls for intended issues/tickets/commits/notion pages 
                                                       from which the answer is supposed to be 
Context:
{context}

Based on the context above,
Question: {input}. For the question do not actually answer the question, just provide the urls for the intended answer. Provide a list of urls or a single url for issue/ticket/commit/notion page if applicable otherwise return an empty list. The list should be like the following format:
                                                       ["url1","url2"]
""")
document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
retriever_body = vector_store.as_retriever()
retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)

document_chain_url = create_stuff_documents_chain(llm, prompt_template_url)
retriever_url = vector_store.as_retriever()
retrieval_chain_url = create_retrieval_chain(retriever_url, document_chain_url)

def query_assistant_body(input):
    response = retrieval_chain_body.invoke({"input": input})
    context=response.get("context","No context available")
    print(context)
    return response["answer"]

def query_assistant_url(input):
    response = retrieval_chain_url.invoke({"input": input})
    return response["answer"]

user_input = st.text_input("Ask your question:", "")
if st.button("Send"):
    if user_input:
        answer = query_assistant_body(user_input)
        urls=query_assistant_url(user_input)
        urls=json.loads(urls)
        links = []
        for url in urls:
            links.append(f"[{url}]({url})\n")
        answer = answer.replace(",","\n")
        st.text_area("Answer:", value=answer, height=300, key="bot_response")
        if links:
            st.markdown("### Related Links")
            st.markdown("\n".join(links))
        else:
            st.markdown("### Related Links")
            st.markdown("No relevant links found.")
        