import re
import os
from dotenv import load_dotenv

load_dotenv()

TITLE = os.getenv("DOC_TITLE")

def clean_text(text):
    text = re.sub(r'\n\n\n+', '\n\n\n', text)          # multiple newlines
    text = re.sub(r'\s+', ' ', text)                   # whitespace
    text = text.strip()
    return text


def remove_footer(text, page_num):

    escaped_title = re.escape(TITLE.strip())

    pattern1 = rf"\s*{page_num}\s+{escaped_title}\s*$"
    pattern2 = rf"\s*{escaped_title}\s+{page_num}\s*$"

    text = re.sub(pattern1, '', text)
    text = re.sub(pattern2, '', text)
    
    return text.rstrip()