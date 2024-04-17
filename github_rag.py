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



def load_corpus(file_path='github_corpus.json'):
    with open(file_path, 'r',encoding="utf-8") as f:
        return json.load(f)


def get_vector_store(text, embeddings):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def flatten_repo_data(all_repos_data):
    lines = [] 

    for repo_data in all_repos_data:
        repo_name = f"repo_name:{repo_data['name']}"
        repo_description = f"repo_description:{repo_data['description']}"

        # Handle issues for the repository
        for issue in repo_data['issues']:
            issue_body_cleaned = issue['body'].replace('\n', ' ')
            issue_line = [
                "Issue:",
                repo_name,
                repo_description,
                f"issue_title:{issue['title']}",
                f"issue_number:{issue['number']}",
                f"issue_state:{issue['state']}",
                f"created_at:{issue['created_at']}",
                f"updated_at:{issue['updated_at']}",
            ]
            lines.append(','.join(issue_line))

        for branch in repo_data['branches']:
            branch_name = f"branch_name:{branch['name']}"
            for commit in branch['commits']:
                commit_message_cleaned = commit['message'].replace('\n', ' ')
                commit_line = [
                    "Commit:",
                    repo_name,
                    repo_description,
                    branch_name,
                    f"commit_message:{commit_message_cleaned}",
                    f"commit_date:{commit['date']}",
                    f"commit_author:{commit['author']}"
                ]
                lines.append(','.join(commit_line))
    text = "\n".join(lines)
    return text


corpus = load_corpus()
text=flatten_repo_data(corpus)

embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
vector_store = get_vector_store(text, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. When a user refers to an issue they will be labelled in context by Issue:. When a user refers to a commit 
they will be labelled by Commit:.The issues witll have a repo_name key which will connect them to a repository .The commits will have a branch_name key which will connect them to a branch and a repo_name
key which will connect them to a repository. Only answer whatever information the user asks do not give whole issues or commits as response.
                                        


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
    st.title("GITHUB ASSISTANT")
    user_input = st.text_input("Ask your question:", "")
    if st.button("Send"):
        if user_input:
            answer = query_assistant(user_input)
            st.text_area("Answer:", value=answer, height=100, key="bot_response")

if __name__ == "__main__":
    main()
