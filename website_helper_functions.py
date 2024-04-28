import os








def custom_chunking_website(directory='website_data'):
    chunks = []
    arbitrary_strings_to_remove = ["Copyright Policy", "California Privacy Disclosure", "Privacy Policy", "Learn More", "Services"]

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
            url = 'https://' + filename.replace('_', '/').replace('.txt', '')
            modified_lines = [url + '\n'] + ["Website: " + line for line in lines if line.strip() not in arbitrary_strings_to_remove]
            chunk = ''.join(modified_lines)
            chunks.append(chunk)
            text='\n'.join(chunks)
    return text