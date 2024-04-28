import solara
from solara.components.file_drop import FileInfo
from ai_assistant_chain import extract_urls, load_pdf_vector, load_vector_store
from pathlib import Path
from typing import Optional, cast,List
import textwrap
import os






loader_visible = solara.reactive(True)
input_message=solara.reactive("")
output_message=solara.reactive("")
output_urls=solara.reactive("")
notion_checkbox = solara.reactive(False)
jira_checkbox = solara.reactive(False)
github_checkbox = solara.reactive(False)
website_checkbox=solara.reactive(True)
file_upload_checkbox = solara.reactive(False)






global retrieval_chain_body
retrieval_chain_body=load_vector_store('Website')
print("Vector Store all loaded")
def query_assistant_body(input):
    global retrieval_chain_body
    response = retrieval_chain_body.invoke({"input": input})
    context=response.get("context","No context available")
    # print(context)
    context_text=""
    for document in context:
        context_text+=document.page_content
    urls=extract_urls(context_text)
    # print(urls)
    return response["answer"],urls

def handle_update(new_value):
    response,urls=query_assistant_body(new_value)
    url_markdown="Related Links: <br />"
    for url in urls:
        url_markdown+=f"[{url}]({url})<br />"
    output_urls.value=url_markdown
    response=response.replace("\n","<br />")
    output_message.value=response


def on_value_change_tools(value,name):
    print(value)
    print(name)
    global retrieval_chain_body
    if value:
        file_upload_checkbox.value = False
    if value and name=="Notion":
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False
    if value and name=="Jira":
        notion_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False
    if value and name=="Github":
        notion_checkbox.value = False
        jira_checkbox.value = False
        website_checkbox.value = False
    if value and name=="Website":
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False



def on_value_change_file(value):
    if value:
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False

@solara.component
def Page():
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
                                              on_value=lambda value: on_value_change_tools(value,"Notion"))
            checkbox_jira = solara.Checkbox(label="Jira", value=jira_checkbox,
                                            on_value=lambda value: on_value_change_tools(value,"Jira"))
            checkbox_github = solara.Checkbox(label="Github", value=github_checkbox,
                                              on_value=lambda value: on_value_change_tools(value,"Github"))
            checkbox_website = solara.Checkbox(label="Website", value=website_checkbox,
                                              on_value=lambda value: on_value_change_tools(value,"Website"))
            checkbox_file = solara.Checkbox(label="File Upload", value=file_upload_checkbox,
                                            on_value=lambda value: on_value_change_file(value))
        solara.InputText("Type your message", value=input_message,on_value=handle_update,update_events=["keyup.enter"])
        if loader_visible.value:
            solara.ProgressLinear(True)
        if file_upload_checkbox.value:
            solara.FileDrop(label="Drag and drop a pdf.",on_file=on_file,lazy=True)
        solara.Markdown(output_message.value)
        solara.Markdown(output_urls.value)