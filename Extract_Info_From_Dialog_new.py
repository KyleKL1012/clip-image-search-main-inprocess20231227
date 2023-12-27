import os
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams

api_key = "pak-AwkYTgLft5uB85xVSeqWdZIHaMfWD-VlWnxWC_WU50g"
api_url = "https://bam-api.res.ibm.com/v1/"
creds = Credentials(api_key=api_key, api_endpoint=api_url)


def call_llm_directly(input, creds=creds, model_name="meta-llama/llama-2-70b-chat", decoding_method="sample", max_new_tokens=2000, temperature=0.1):
    params = GenerateParams(decoding_method=decoding_method, max_new_tokens=max_new_tokens, temperature=temperature)
    model_name = model_name
    llm_model = Model(model=model_name, params=params, credentials=creds)
    generated_text = ''
    prompt1 = f"Now you are a lost and found help desk assistant working at an airport. Your job role is to summarize and extract the key information from the dialog, such as what is the lost item, and where did the passenger lose it, or some other key information you would consider important \
            To generate your final response, you need to refer to the information in the following text provided: {input}.\
            Then, please answer my question using the following template:\
            The lost item is: \
            The location where the passenger lost his/her item is: \
            Other important information: "
    for response in llm_model.generate([prompt1]):
        generated_text += response.generated_text
    return generated_text


def get_itemdesc(input, creds=creds, model_name="meta-llama/llama-2-70b-chat", decoding_method="sample", max_new_tokens=100, temperature=0.1):
    params = GenerateParams(decoding_method=decoding_method, max_new_tokens=max_new_tokens, temperature=temperature)
    model_name = model_name
    llm_model = Model(model=model_name, params=params, credentials=creds)
    generated_text = ''
    template = 'The lost item is:'
    summarization1 = call_llm_directly(input,creds=creds)
    prompt2 = f"Imagine you are a Lost and Found help desk assistant at an airport. Your colleague has summarized a dialog recording in {summarization1}. \
            Read the content carefully and accurately extract entities, focusing solely on the lost item. Follow the provided template: {template} for the answer, limiting it to 50 words. \
            Do not repeat the template. Do not include other information such as location, or passenger's name, or passenger's contact information."
    for response in llm_model.generate([prompt2]):
        generated_text += response.generated_text
    return generated_text


