from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def get_llm(model_id):

    text_gen = pipeline(
        'text-generation',
        model=model_id,
        max_new_tokens=250,
        temperature=0.5,
    )

    llm = HuggingFacePipeline(pipeline=text_gen)
    return llm
