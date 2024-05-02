# Import necessary libraries
import solara
from solara.components.file_drop import FileInfo
from ai_assistant_chain import extract_urls, load_pdf_vector, load_vector_store, switch_llm
from pathlib import Path
import reacton.ipyvuetify as v
from typing import Optional, List
import textwrap
import os
import threading

# Initialize reactive variables
notion_checkbox = solara.reactive(True)
jira_checkbox = solara.reactive(True)
github_checkbox = solara.reactive(True)
website_checkbox = solara.reactive(True)
file_upload_checkbox = solara.reactive(False)
llm = solara.reactive("gemini")


# Load LLM with default set to gemini-pro and generate retrieval chain using it, using all sources by default
switch_llm(llm.value, "gemini-pro")
global retrieval_chain_body
retrieval_chain_body = load_vector_store('all', ['notion', 'jira', 'github', 'website'])

def on_value_change_tools(value, name):
    """
    Callback function to handle changes in checkbox values for different tools.

    Parameters:
        value (bool): The new value of the checkbox.
        name (str): The name of the tool associated with the checkbox.
    """
    
    # Initialize an empty list to store selected sources
    sourceList = []  
    
    # Access the global variable 'retrieval_chain_body'
    global retrieval_chain_body  
    
    # If the file upload checkbox is checked, uncheck it
    if value:  
        file_upload_checkbox.value = False  
    
    # Check if each tool checkbox is checked, and add its name to the source list if checked
    if notion_checkbox.value:  
        sourceList.append("notion")  
    if jira_checkbox.value:  
        sourceList.append("jira")  
    if github_checkbox.value:  
        sourceList.append("github")  
    if website_checkbox.value:  
        sourceList.append("website")  
    
    # Load the vector store for all selected sources
    retrieval_chain_body = load_vector_store('all', sourceList)

def on_value_change_llm(value):
    """
    Callback function to handle changes in the LLM (Language Learning Model) selection.

    Parameters:
        value (str): The new value of the LLM.
    """
    # Access the global variable 'retrieval_chain_body'
    global retrieval_chain_body
    
    # If the selected value is "gemini", switch LLM to "gemini-pro"
    if value == "gemini":
        switch_llm("gemini", "gemini-pro")
    else:
        # Depending on the selected value, switch LLM accordingly
        if value == "gemma-7b-it":
            switch_llm("llama", "gemma-7b-it")
        elif value == "llama3-8b-8192":
            switch_llm("llama", "llama3-8b-8192")
        elif value == "llama3-70b-8192":
            switch_llm("llama", "llama3-70b-8192")
        elif value == "mixtral-8x7b-32768":
            switch_llm("llama", "mixtral-8x7b-32768")
    
    # Initialize an empty list to store selected sources
    sourceList = []
    
    # Check if each tool checkbox is checked, and add its name to the source list if checked
    if notion_checkbox.value:
        sourceList.append("notion")
    if jira_checkbox.value:
        sourceList.append("jira")
    if github_checkbox.value:
        sourceList.append("github")
    if website_checkbox.value:
        sourceList.append("website")
    
    # Load the vector store for all selected sources
    retrieval_chain_body = load_vector_store('all', sourceList)

def on_value_change_file(value):
    """
    Callback function to handle changes in the file upload checkbox.

    Parameters:
        value (bool): The new value of the file upload checkbox.
    """
    # If the file upload checkbox is checked
    if value:
        # Uncheck all other checkboxes related to tool selection
        notion_checkbox.value = False
        jira_checkbox.value = False
        github_checkbox.value = False
        website_checkbox.value = False

image_url="logo2.png"
@solara.component
def Page():
    """
    Solara component for creating an AI Assistant page.

    This component includes functionalities to interact with an AI assistant,
    handle user queries, display instructions, and manage tools and file uploads.

    """

    # List of available language models
    llms = ["gemini", "gemma-7b-it", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]

    # Set the title of the page
    solara.Title("AI Assistant")
    solara.Image(image_url,width='300px')

    # Initialize state variables for managing UI state
    loader, set_loader = solara.use_state(False)
    input_message, set_input_message = solara.use_state("")
    output_message, set_output_message = solara.use_state("")
    output_urls, set_output_urls = solara.use_state("")
    processing, set_processing = solara.use_state(False)

    def query_assistant_body(input):
        """
        Query the AI assistant with the provided input and update the UI accordingly.

        Args:
            input (str): The user input query.

        """
        # Normalize input to lowercase
        input = input.lower()

        # Set processing state to true
        set_processing(True)

        # Invoke the retrieval chain to get the assistant's response
        global retrieval_chain_body
        response = retrieval_chain_body.invoke({"input": input})

        # Extract context from the response
        context = response.get("context", "No context available")
        context_text = ""
        
        # Process context to extract relevant information
        if context:
            if not file_upload_checkbox.value:
                if context[0].metadata["source"] == "notion":
                    context_text = context[0].page_content + "\n" + context[1].page_content
                else:
                    for document in context:
                        if document.metadata["source"] != "notion":
                            context_text += document.page_content
            else:
                for document in context:
                    context_text=document.page_content

            # Write context to file
            with open('context.txt', "w", encoding="utf-8") as f:
                f.write(context_text)

        # Extract URLs from context
        urls = extract_urls(context_text)

        # Format URLs for display
        url_markdown = "Related Links: <br />"
        for url in urls:
            if "notion" in url and len(url) < 56:
                continue
            else:
                url_markdown += f"[{url}]({url})<br />"

        # Update output URLs state
        set_output_urls(url_markdown)

        # Process assistant response
        if response["answer"] == "no context":
            if len(urls) > 0:
                response["answer"] = "I do not have the knowledge for what you have asked but the following links may prove helpful"
            else:
                response["answer"] = "Sorry, I can not help you with your query as I do not have the knowledge"

        # Format response for display
        response = response["answer"].replace("\n", "<br />")

        # Update output message state
        set_output_message(response)

        # Set loader and processing states to false
        set_loader(False)
        set_processing(False)

    def handle_update(*ignore_args):
        """
        Handle updates to the input text area.

        This function is triggered when the user presses enter or clicks the Send Button.

        """
        if not processing:
            # Set loader state to true
            set_loader(True)

            # Start a new thread to query the assistant with user input
            thread = threading.Thread(target=query_assistant_body, args=(input_message,))
            thread.start()

    def on_file(f: FileInfo):
        """
        Handle file uploads and load vector store for uploaded PDF.

        Args:
            f (FileInfo): Information about the uploaded file.

        """
        global retrieval_chain_body

        # Set filename, size, and content states
        set_filename(f["name"])
        set_size(f["size"])
        content = f["file_obj"].read(f["size"])
        set_content(content)

        # Write file content to disk
        file_path = f["name"]
        with open(file_path, 'wb') as file:
            file.write(content)

        # Load PDF vector store
        retrieval_chain_body = load_pdf_vector(file_path)

    # Initialize state variables for file upload
    content, set_content = solara.use_state(b"")
    filename, set_filename = solara.use_state("")
    size, set_size = solara.use_state(0)

    # Define the main layout
    with solara.Column() as main:
        with solara.Sidebar():
            # Select language model
            solara.Select(label="Select Model", value=llm, values=llms, on_value=on_value_change_llm)

            # Checkboxes for tools
            solara.Switch(label="Notion", value=notion_checkbox,
                                            on_value=lambda value: on_value_change_tools(value, "notion"))
            solara.Switch(label="Jira", value=jira_checkbox,
                                          on_value=lambda value: on_value_change_tools(value, "jira"))
            solara.Switch(label="Github", value=github_checkbox,
                                            on_value=lambda value: on_value_change_tools(value, "github"))
            solara.Switch(label="Website", value=website_checkbox,
                                             on_value=lambda value: on_value_change_tools(value, "website"))
            solara.Switch(label="File Upload", value=file_upload_checkbox,
                                          on_value=lambda value: on_value_change_file(value))

        # Define the main content area
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

        # Text input field
        text_field = v.TextField(v_model=input_message, on_v_model=set_input_message, label="Enter your message")
        v.use_event(text_field, "keydown.enter", handle_update)

        # Send button
        solara.Button(label="Send", color="primary", on_click=handle_update)

        # Loader
        solara.ProgressLinear(loader)

        # File upload component
        if file_upload_checkbox.value:
            solara.FileDrop(label="Drag and drop a pdf.", on_file=on_file, lazy=True)

        # Output message
        solara.Markdown(output_message)

        # Output URLs
        solara.Markdown(output_urls)
