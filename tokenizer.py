import re

def tokenize(text):
    # Semplice tokenizzazione usando regex
    tokens = re.findall(r'\b\w+\b', text)
    return tokens
