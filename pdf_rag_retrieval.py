from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from load_env import openAIKey

# Embeddings Open AI model
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openAIKey
)

# Instantiate an retriever
retriever = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# Create vector embeddngs of the user query,
# and returns the relevant vector embeddings fro the database

def getRelevantChunks(input: str):
    search_result = retriever.similarity_search(
        query=input
    )

    page_content_list = []

    for chunk in search_result:
        metadata = chunk.metadata
        page_content = chunk.page_content
        page_number = metadata.get('page')
        page_content_dict = {"page_number": page_number, "page_content": page_content}
        page_content_list.append(page_content_dict)


    return page_content_list

