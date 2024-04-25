import os
import json



keys_to_remove = [
    "id", "color", "type", "link", "href", "public_url", "object",
    "database_id", "icon", "cover", "bold", "italic", "strikethrough",
    "underline", "code", "archived", "last_edited_time", "created_by",
     "relation", "has_more", "Sub-item","has_children","last_edited_by","annotations","is_toggleable","divider","toggle","file",'created_time',"unsupported"

]


    
def remove_keys_from_dict(d, keys):
    if isinstance(d, dict):
        return {k: (remove_keys_from_dict(v, keys) if k != 'page_id' else f"https://www.notion.so/bytecorp/{v.replace('-', '')}")
                for k, v in d.items() if k not in keys}
    elif isinstance(d, list):
        return [remove_keys_from_dict(v, keys) for v in d]
    else:
        return d
    

def parse_value(v, new_key):
    if isinstance(v, dict):
        return parse_dict(v, new_key) if v else None
    elif isinstance(v, list):
        processed_items = [item for item in map(lambda x: parse_value(x, ''), v) if item]
        return f"{new_key}:[{','.join(processed_items)}]" if processed_items else None
    elif v:  
        return f"{new_key}:{v}"
    return None

def parse_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}:{k}" if parent_key else k
        item = parse_value(v, new_key)
        if item:
            items.append(item)
    return ','.join(filter(None, items))





