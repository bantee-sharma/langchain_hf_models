from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    huggingfacehub_api_token=huggingface_api_key
) 

model = ChatHuggingFace(llm = llm)

result = model.invoke("who is rohit sharma")

print(result)