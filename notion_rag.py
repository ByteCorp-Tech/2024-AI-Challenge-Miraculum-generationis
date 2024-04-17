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


keys_to_remove = [
    "id", "color", "type", "link", "href", "public_url", "object",
    "database_id", "icon", "cover", "bold", "italic", "strikethrough",
    "underline", "code", "archived", "last_edited_time", "created_by",
    "parent", "relation", "has_more", "Sub-item","has_children","last_edited_by","annotations","is_toggleable",
    "url","divider","toggle","file",'created_time',"unsupported"

]

def load_corpus(file_path='notion_corpus.json'):
    with open(file_path, 'r',encoding="utf-8") as f:
        return json.load(f)
    
def remove_keys_from_dict(d, keys):
    if isinstance(d, dict):
        return {k: remove_keys_from_dict(v, keys) for k, v in d.items() if k not in keys}
    elif isinstance(d, list):
        return [remove_keys_from_dict(v, keys) for v in d]
    else:
        return d
    

def parse_value(v, new_key):
    if isinstance(v, dict):
        # Only parse non-empty dictionaries
        return parse_dict(v, new_key) if v else None
    elif isinstance(v, list):
        # Process each item in the list, removing empty ones
        processed_items = [item for item in map(lambda x: parse_value(x, ''), v) if item]
        # Join the non-empty items with commas and only return if there's something to show
        return f"{new_key}:[{','.join(processed_items)}]" if processed_items else None
    elif v:  # Check if value is not an empty string
        return f"{new_key}:{v}"
    # Exclude empty strings, empty lists, and empty dicts
    return None

def parse_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}:{k}" if parent_key else k
        item = parse_value(v, new_key)
        if item:
            items.append(item)
    return ','.join(filter(None, items))

def get_vector_store(text, embeddings):
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase



corpus = load_corpus()
cleaned_data = remove_keys_from_dict(corpus, keys_to_remove)
formatted_data = [parse_dict(page) for page in cleaned_data]
text=""

for data in formatted_data:
    text=text+data+"\n"



embeddings = OpenAIEmbeddings()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
vector_store = get_vector_store(text, embeddings)


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
    st.title("NOTION ASSISTANT")
    user_input = st.text_input("Ask your question:", "")
    if st.button("Send"):
        if user_input:
            answer = query_assistant(user_input)
            st.text_area("Answer:", value=answer, height=100, key="bot_response")

if __name__ == "__main__":
    main()
