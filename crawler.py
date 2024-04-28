import os
from langchain.llms import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

def load_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

prompt_template_body = ChatPromptTemplate.from_template("""Answer the questions in full detail based on the context provided. Reply with a simple string "no context" if you cannot answer the question.
Context:
{context}

Remember your job is to use the whole context for answers and do not leave any detail and do not give vague short answers. That is your primary responsibility.
                                                                                                     Reply with "no context" if you can not answer the question from the context                                                               

                                                        . 
Question: {input}. Based on the context above answer the question.Use the whole context for answers and answer in full detail.

                    If the question can not be answered by context and you cant answer any question then just reply "no context".
""")





def custom_chunking_website(directory='website_data'):
    chunks = []
    arbitrary_strings_to_remove = ["Copyright Policy", "California Privacy Disclosure", "Privacy Policy","Learn More","Services"]

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r',encoding="utf-8") as file:
                lines = file.readlines()
            url = 'https://' + filename.replace('_', '/').replace('.txt', '')
            modified_lines = [url + '\n'] + lines
            modified_lines = [line for line in modified_lines if line.strip() not in arbitrary_strings_to_remove]
            chunk = ''.join(modified_lines)
            # print(chunk)
            chunks.append(chunk)
    return chunks




def get_vector_store_website():
        embeddings = OpenAIEmbeddings()
        chunks=custom_chunking_website('website_data')
        vector_store = Chroma.from_texts(chunks, embeddings)
        document_chain_body = create_stuff_documents_chain(llm, prompt_template_body)
        retriever_body = vector_store.as_retriever()
        retrieval_chain_body = create_retrieval_chain(retriever_body, document_chain_body)
        return retrieval_chain_body








def gen_answer(question):
    retriever = get_vector_store_website()
    response=retriever.invoke({"input": question})
    print(response["answer"])

gen_answer("Tell me our team members and board of advisors")






    
    