import os
from dotenv import load_dotenv
import pickle

from pdf_chunker import extract_with_metadata


with open("data/chunks_metadata.pkl", 'rb') as file:
    documents = pickle.load(file)
