from openai import OpenAI
from load_env import openAIKey;
from pdf_rag_retrieval import getRelevantChunks
from pdf_rag_ingestion import dataIngestion
from system_prompt import getSystemPrompt
import json

## Perform the Data ingestion
# dataIngestion()

client = OpenAI(
    api_key=openAIKey
)

## Read the user query
user_query = input("User: > ")

## Retrieve the relevant embeddings/chunks from the datastore
system_context = getRelevantChunks(user_query)

## Create the system prompt for the LLM
SYSTEM_PROMPT = getSystemPrompt(json.dumps(system_context))

response = client.responses.create(
    model="gpt-4o",
    instructions=SYSTEM_PROMPT,
    input=user_query
)

print(response.output_text)


