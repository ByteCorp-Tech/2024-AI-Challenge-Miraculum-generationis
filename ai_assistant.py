import streamlit as st
import json
from dotenv import load_dotenv
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from github_helper_functions import flatten_repo_data
from jira_helper_functions import flatten_corpus
from notion_helper_functions import parse_dict, remove_keys_from_dict,keys_to_remove
import langchain

load_dotenv()

# Load and process Notion corpus
def load_corpus(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)



notion_corpus = load_corpus('corpus/notion_corpus.json')
notion_cleaned = remove_keys_from_dict(notion_corpus, keys_to_remove)
notion_text = '\n'.join([parse_dict(page) for page in notion_cleaned])

jira_corpus = load_corpus('corpus/jira_corpus.json')
jira_text = flatten_corpus(jira_corpus)


github_corpus = load_corpus('corpus/github_corpus.json')
github_text = flatten_repo_data(github_corpus)





# Streamlit interface setup
st.title("AI Assistant")
toggle_notion = st.sidebar.checkbox("Notion", True)
toggle_jira = st.sidebar.checkbox("JIRA", True)
toggle_github = st.sidebar.checkbox("GitHub", True)
show_context = st.sidebar.checkbox("Show Source", False)

active_texts = []
if toggle_notion:
    active_texts.append(notion_text)
if toggle_jira:
    active_texts.append(jira_text)
if toggle_github:
    active_texts.append(github_text)

combined_text = "\n".join(active_texts)

embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=100, length_function=len)
chunks = text_splitter.split_text(combined_text)
vector_store = FAISS.from_texts(chunks, embeddings)

prompt_template = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. For every information you provide in answer, if theres a 
                                                   link/url with it provide it in the answer 
Context:
{context}

Based on the context above,
Question: {input}. provide urls separately in answer too for every commit/issue/ticket/page you pull information from. Output in a json format like this:
                                                   "body":"(Complete body of the answer and if there are key/value pairs each key/value pair should be separated by commas)","url":"(a list of urls for commit/issue/ticket/page of the answer. if answer is from single source then provide one url)"
""")

document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def query_assistant(input):
    response = retrieval_chain.invoke({"input": input})
    context = response.get("context", "No context available.")
    return response["answer"], context

user_input = st.text_input("Ask your question:", "")
if st.button("Send"):
    if user_input:
        answer, context = query_assistant(user_input)
        st.text_area("Answer:", value=answer, height=100, key="bot_response")
        if show_context:
            for doc in context:
                print(doc)
            st.markdown("### Source")
            st.write(context)  # Properly format and display context here