from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    task = 'text-classification',
    
)


res = llm.invoke('that movie was great')

print(res.content)