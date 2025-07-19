from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def document_loader(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


def doc_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)