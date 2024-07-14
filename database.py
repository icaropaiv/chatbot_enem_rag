from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
import pdfplumber


def extract_file_content(document):
    raw_text = ""

    with pdfplumber.open(document) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Remove espaços extras dentro das palavras
                text = ' '.join(text.split())
                raw_text += text + '\n'

    return raw_text

def create_database():

    data = extract_file_content('enem_2024.pdf')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_text(data)
    print(chunks)

    embeddings=OpenAIEmbeddings(
        model="text-embedding-ada-002", 
        chunk_size=1000, 
        openai_api_key="OPENAI_API_KEY")
    

    db = FAISS.from_texts(chunks, embeddings)

    return db
