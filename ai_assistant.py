import solara
from solara.components.file_drop import FileInfo
from ai_assistant_chain import extract_urls, load_pdf_vector, load_vector_store,switch_llm
from pathlib import Path
import reacton.ipyvuetify as v
from typing import Optional, cast,List
import textwrap
import os
import threading







notion_checkbox = solara.reactive(True)
jira_checkbox = solara.reactive(True)
github_checkbox = solara.reactive(True)
website_checkbox=solara.reactive(True)
file_upload_checkbox = solara.reactive(False)
llm=solara.reactive("gemini")
switch_llm(llm.value,"gemini-pro")
global retrieval_chain_body
retrieval_chain_body=load_vector_store('all',['notion','jira','github','website'])






def on_value_change_tools(value,name):
    print(value)
    print(name)
    sourceList=[]
    global retrieval_chain_body
    if value:
        file_upload_checkbox.value = False
    if notion_checkbox.value:
        sourceList.append("notion")
    if jira_checkbox.value:
        sourceList.append("jira")
    if github_checkbox.value:
        sourceList.append("github")
    if website_checkbox.value:
        sourceList.append("website")
    retrieval_chain_body=load_vector_store('all',sourceList)

def on_value_change_llm(value):
    global retrieval_chain_body
    if value=="gemini":
        switch_llm("gemini","gemini-pro")
    else:
        if value=="gemma-7b-it":
            switch_llm("llama","gemma-7b-it")
        elif value=="llama3-8b-8192":
            switch_llm("llama","llama3-8b-8192")
        elif value=="llama3-70b-8192":
            switch_llm("llama","llama3-70b-8192")
        elif value=="mixtral-8x7b-32768":
            switch_llm("llama","mixtral-8x7b-32768")
    sourceList=[]
    if notion_checkbox.value:
        sourceList.append("notion")
    if jira_checkbox.value:
        sourceList.append("jira")
    if github_checkbox.value:
        sourceList.append("github")
    if website_checkbox.value:
        sourceList.append("website")
    retrieval_chain_body=load_vector_store('all',sourceList)



def on_value_change_file(value):
    if value:
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False

@solara.component
def Page():
    llms=["gemini","gemma-7b-it","llama3-8b-8192","llama3-70b-8192","mixtral-8x7b-32768"]
    
    solara.Title("AI Assistant")
    loader,set_loader=solara.use_state(False)
    input_message,set_input_message=solara.use_state("")
    output_message, set_output_message = solara.use_state("")
    output_urls,set_output_urls=solara.use_state("")
    processing, set_processing = solara.use_state(False)

    def query_assistant_body(input):
        input=input.lower()
        set_processing(True)
        global retrieval_chain_body
        response = retrieval_chain_body.invoke({"input": input})
        context=response.get("context","No context available")
        context_text=""
        if context[0].metadata["source"]=="notion":
            context_text=context[0].page_content+"\n"+context[1].page_content
        else:   
            for document in context:
                if document.metadata["source"]!="notion":
                    context_text+=document.page_content
        with open('context.txt',"w",encoding="utf-8") as f:
            f.write(context_text)
        urls=extract_urls(context_text)
        response=response["answer"]
        url_markdown="Related Links: <br />"
        for url in urls:
            if "notion" in url and len(url)<56:
                continue
            else:
                url_markdown+=f"[{url}]({url})<br />"
        set_output_urls(url_markdown)
        if response=="no context":
            if len(urls)>0:
                response="I do not have the knowledge for what you have asked but the following links may prove helpful"
            else:
                response="Sorry, I can not help you with your query as I do not have the knowledge"
        response=response.replace("\n","<br />")
        set_output_message(response)
        set_loader(False)
        set_processing(False)
        # 


    def handle_update(*ignore_args):
        if not processing:  # Only proceed if not currently processing
            set_loader(True)
            thread = threading.Thread(target=query_assistant_body, args=(input_message,))
            thread.start()
        


    def on_file(f: FileInfo):
        global retrieval_chain_body
        set_filename(f["name"])
        set_size(f["size"])
        content = f["file_obj"].read(f["size"])
        set_content(content)  
        file_path = f["name"]
        with open(file_path, 'wb') as file:
            file.write(content)
        retrieval_chain_body=load_pdf_vector(file_path)

    
    content, set_content = solara.use_state(b"")
    filename, set_filename = solara.use_state("")
    size, set_size = solara.use_state(0)
    with solara.Column() as main:
        with solara.Sidebar():
            solara.Select(label="Select Model",value=llm,values=llms,on_value=on_value_change_llm)
            
            with solara.Card(title="Instructions"):
                solara.Markdown(""" **To help you get the best answers, please provide detailed queries.**
**For Instance:**<br />
Instead of: "Information X"<br />
Try: "Find me Information X"<br />
**Be specific about where to find the information you need:**<br />
Instead of: "Who is Person X"<br />
Try: "Who is Person X in Team Y"<br />
By guiding your queries, you enable the assistant to understand and respond more effectively.
  """)
            
            checkbox_notion = solara.Checkbox(label="Notion", value=notion_checkbox,
                                              on_value=lambda value: on_value_change_tools(value,"notion"))
            checkbox_jira = solara.Checkbox(label="Jira", value=jira_checkbox,
                                            on_value=lambda value: on_value_change_tools(value,"jira"))
            checkbox_github = solara.Checkbox(label="Github", value=github_checkbox,
                                              on_value=lambda value: on_value_change_tools(value,"github"))
            checkbox_website = solara.Checkbox(label="Website", value=website_checkbox,
                                              on_value=lambda value: on_value_change_tools(value,"website"))
            checkbox_file = solara.Checkbox(label="File Upload", value=file_upload_checkbox,
                                            on_value=lambda value: on_value_change_file(value))
       
        text_field = v.TextField(v_model=input_message, on_v_model=set_input_message, label="Enter your message")
        v.use_event(text_field, "keydown.enter", handle_update)
        solara.Button(label="Send", color="primary",on_click=handle_update)

        solara.ProgressLinear(loader)
        if file_upload_checkbox.value:
            solara.FileDrop(label="Drag and drop a pdf.",on_file=on_file,lazy=True)
        solara.Markdown(output_message)
        solara.Markdown(output_urls)