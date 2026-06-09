def clean_content(content: str):
    if content is None:
        raise ValueError("content is None.")
    
    content = content.strip()
    
    if content.startswith('\"') and content.endswith('\"'):
        content = content[1:-1]
    
    if content.startswith("```\n") and content.endswith("\n```"):
        content = content[4:-4]
    
    return content

