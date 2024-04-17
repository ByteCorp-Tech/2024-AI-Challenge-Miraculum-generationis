from notion_client import Client
import os
from dotenv import load_dotenv
load_dotenv()
NOTION_API_KEY=os.getenv("NOTION_API_KEY")
notion = Client(auth=NOTION_API_KEY)
import json


def fetch_block_children(block_id, notion):
    """Fetch all children blocks of a given block and include all block data."""
    block_children = notion.blocks.children.list(block_id=block_id)["results"]
    content = []
    for block in block_children:
        content.append(block)
        if "has_children" in block and block["has_children"]:
            block["children"] = fetch_block_children(block["id"], notion)
    return content

def fetch_all_pages(notion):
    """Fetch all standalone pages in the workspace and include all page and block data."""
    pages = []
    query_results = notion.search(filter={"value": "page", "property": "object"})["results"]

    for page in query_results:
        page_details = {"page_data": page}  
        page_id = page["id"]
        page_details["content"] = fetch_block_children(page_id, notion)  
        pages.append(page_details)

    return pages

def create_corpus(notion):
    """Create a corpus from all standalone pages in the workspace, including all data."""
    corpus = fetch_all_pages(notion)
    return corpus


corpus = create_corpus(notion)

# Save the corpus to a JSON file
with open("corpus/notion_corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=4)