from langchain_core.prompts import ChatPromptTemplate


prompt_template_body = ChatPromptTemplate.from_template("""Answer the questions based on the context provided. 
Context:
{context}

Based on the context above answer the question.You are not supposed to answer outside of the context in any condition as this will have serious repurcussions. Just reply that you dont know if the context does not
                                                        the answer to the question
Question: {input}.
""")


prompt_template_url = ChatPromptTemplate.from_template("""You are a url extractor. Whenever a user asks a question your job not to answer the question but
                                                       to extract the urls for the intended answer. Extract urls for intended issues/tickets/commits/notion pages 
                                                       from which the answer is supposed to be 
Context:
{context}

Based on the context above,
Question: {input}. For the question do not actually answer the question, just provide the urls for the intended answer.
                                                        Provide a list of urls or a single url for issue/ticket/commit/notion page if applicable otherwise return an empty list.
                                                        The list should be like the following format:
                                                       ["url1","url2"]
""")
