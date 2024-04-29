import solara
from solara.components.file_drop import FileInfo
from ai_assistant_chain import extract_urls, load_pdf_vector, load_vector_store
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






global retrieval_chain_body
retrieval_chain_body=load_vector_store('all',['notion','jira','github','website'])
print("Vector Store Website loaded")




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



def on_value_change_file(value):
    if value:
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False

@solara.component
def Page():
    solara.Title("AI Assistant")
    loader,set_loader=solara.use_state(False)
    input_message,set_input_message=solara.use_state("")
    output_message, set_output_message = solara.use_state("")
    output_urls,set_output_urls=solara.use_state("")
    processing, set_processing = solara.use_state(False)

    def query_assistant_body(input):
        set_processing(True)
        global retrieval_chain_body
        input+=". Team Bytecorp"
        response = retrieval_chain_body.invoke({"input": input})
        context=response.get("context","No context available")    
        context_text=""
        for document in context:
            context_text+=document.page_content
        with open('context.txt',"w",encoding="utf-8") as f:
            f.write(context_text)
        urls=extract_urls(context_text)
        response=response["answer"]
        url_markdown="Related Links: <br />"
        for url in urls:
            url_markdown+=f"[{url}]({url})<br />"
        set_output_urls(url_markdown)
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