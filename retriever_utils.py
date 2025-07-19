from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def embeddings_model(EMB_MODEL):

    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu" }
    )

def vector_database(chunks, EMB_MODEL):

    emb = embeddings_model(EMB_MODEL)
    vectordb = Chroma.from_documents(chunks, embedding=emb)
    return vectordb

