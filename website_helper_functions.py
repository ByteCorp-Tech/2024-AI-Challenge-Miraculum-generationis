import os








def custom_chunking_website(directory='website_data'):
    chunks = []
    arbitrary_strings_to_remove = ["Copyright Policy", "California Privacy Disclosure", "Privacy Policy","Learn More","Services","Expertise","Staff Augmentation","Other Services","Web Development","Work","Anmol","CreatorVerse",
                                   "DevOps","MLOps","EdTech","Web 3","Design Consultancy","Blockchain Development","AWS","Careers","Company","About Us","Our Team","Board of Advisors","Platforms"
                                   ]

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r',encoding="utf-8") as file:
                lines = file.readlines()
            url = 'https://' + filename.replace('_', '/').replace('.txt', '')
            modified_lines = [url + '\n'] + lines
            modified_lines = [line for line in modified_lines if line.strip() not in arbitrary_strings_to_remove]
            chunk = ''.join(modified_lines)
            chunk='website\n'+chunk
            # print(chunk)
            chunks.append(chunk)
    return chunks