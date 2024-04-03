import streamlit as st
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
load_dotenv()
def load_corpus(file_path='jira_corpus.json'):
    with open(file_path, 'r',encoding="utf-8") as f:
        return json.load(f)

def flatten_corpus(corpus):
    texts = ''
    for project in corpus:
        for issue in project['issues']:
            texts += f"Issue/Ticket: Project:{project['project_name']}, Issue Key: {issue['issue_key']}, Summary: {issue['issue_summary']}, Type: {issue['issue_type']}, Status: {issue['issue_status']}\n"
            for comment in issue['comments']:
                texts += f"Comment: Comment ID: {comment['comment_id']}, Author: {comment['comment_author']}, Body: {comment['comment_body']}, Issue Key: {issue['issue_key']}\n"
    return texts

def get_vector_store(text, embeddings):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

# Load and prepare the corpus
corpus = load_corpus()
text = flatten_corpus(corpus)

# Initialize necessary components
embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
vector_store = get_vector_store(text, embeddings)
prompt = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. When a user refers to an issue or a ticket they will be labelled in context by Issue/Ticket:. When a user refers to a comment 
they will be labelled by Comment:. The comments will also have an issue key which will connect them to a ticket/issue.Format the output According to the Output Format given below.A new line in format will be referenced by
(newline).

Output Format In Case of An Issue/Ticket:
Project Key: (project key)
Issue Key: (issue key)
Issue Summary: (issue summary)
Issue Type: (issue type)
Issue Status: (issue status)

Output Format In Case of A Comment:
Issue Key: (issue key)
Comment ID: (comment id)
Author: (author)
Body: (body)
                                                                                                                                                                                                                                                                                                                                                                                       

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
    st.title("JIRA UI")
    user_input = st.text_input("Ask your question:", "")
    if st.button("Send"):
        if user_input:
            answer = query_assistant(user_input)
            st.text_area("Answer:", value=answer, height=100, key="bot_response")

if __name__ == "__main__":
    main()
