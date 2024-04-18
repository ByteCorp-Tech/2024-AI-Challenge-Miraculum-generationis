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
from langchain_openai.llms import OpenAI
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
st.set_page_config(layout="wide")
st.title("AI Assistant")
toggle_notion = st.sidebar.checkbox("Notion", True)
toggle_jira = st.sidebar.checkbox("JIRA", True)
toggle_github = st.sidebar.checkbox("GitHub", True)

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
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
chunks = text_splitter.split_text(combined_text)
vector_store = FAISS.from_texts(chunks, embeddings)

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
Question: {input}. Provide a list of urls or a single url for issue/ticket/commit/notion page if applicable otherwise return an empty list. The list should be like the following format:
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
        