from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task = "text-generation",
    model_kwargs={"max_length":128},
    temperature=0.7,
    )


prompt = PromptTemplate.from_template("{question}")
model = prompt | llm

res = model.invoke("who is current president of india ")

print(res)