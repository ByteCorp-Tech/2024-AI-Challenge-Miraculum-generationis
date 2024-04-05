from dotenv import load_dotenv
import os
import streamlit as st
import openai
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import json
import langchain
langchain.verbose=True
langchain.debug=True


def load_corpus(file_path='notion_corpus.json'):
    with open(file_path, 'r',encoding="utf-8") as f:
        return json.load(f)







def get_vector_store(text, embeddings):
    text_splitter = CharacterTextSplitter(separator=',', chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase




corpus = load_corpus()
text = json.dumps(corpus, ensure_ascii=False, indent=None)





embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
vector_store = get_vector_store(text, embeddings)
# prompt = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. When a user refers to an issue they will be labelled in context by Issue:. When a user refers to a commit 
# they will be labelled by Commit:.The issues witll have a repo_name key which will connect them to a repository .The commits will have a branch_name key which will connect them to a branch and a repo_name
# key which will connect them to a repository.Format the output According to the Output Format given below.

# Output Format In Case of An Issue:
# Repo Name: (repo name)
# Repo Description: (repo description)                                          
# Issue Title: (issue title)
# issue_number: (issue number)
# issue_state: (issue state)                                          
# created_at: (created at)
# updated_at: (updated at)
# issue_body: (issue body)

                                          
# Output Format In Case of A Commit:
# Repo Name: (repo name)
# repo_description: (repo description)                                          
# branch_name: (branch name)
# commit_message: (commit message)
# commit_date: (commit date)
# commit_author: (commit author)                                          


# Context:
# {context}

                                        
                             
# Based on the context above,
# Question: {input}

# """)
prompt = ChatPromptTemplate.from_template("""Answer the questions based on the context provided.
                                        


Context:
{context}

                                        
                             
Based on the context above,
Question: {input}

""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def query_assistant(input):
    response = retrieval_chain.invoke({"input": input})
    return response["answer"]

def main():
    st.title("NOTION RAG UI")
    user_input = st.text_input("Ask your question:", "")
    if st.button("Send"):
        if user_input:
            answer = query_assistant(user_input)
            st.text_area("Answer:", value=answer, height=100, key="bot_response")

if __name__ == "__main__":
    main()
