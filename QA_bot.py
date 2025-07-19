from langchain.chains import RetrievalQA
import gradio as gr
from get_llm import get_llm
from document_utils import document_loader, doc_splitter
from retriever_utils import vector_database


MODEL_ID = 'Qwen/Qwen3-4B'
EMB_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def retriever(file_path: str):

    docs = document_loader(file_path)
    chunks = doc_splitter(docs)
    vectordb = vector_database(chunks, EMB_MODEL)
    return vectordb.as_retriever()


def retriever_qa(file, query: str):
    llm = get_llm(MODEL_ID)
    retr = retriever(file.name)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retr,
        return_source_documents=False
    )
    result = qa_chain.invoke(query)

    result_str = result['result']

    #pick the text btween the "Helpful Answer:" and the next "\n"
    helpful_answer = result_str.split("Helpful Answer:")[1].split("\n")[0]

    return helpful_answer


rag_application = gr.Interface(
    fn=retriever_qa,
    flagging_mode="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf']),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Document Question Answering Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will answer using the provided document."
)

if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
