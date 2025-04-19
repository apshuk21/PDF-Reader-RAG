from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from load_env import openAIKey

def dataIngestion():

    # PDF document loading
    file_path = Path(__file__).parent / "pdf/NodeJS-2013.pdf"
    loader = PyPDFLoader(file_path)

    document = loader.load()

    # PDF splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(document)

    # Embeddings Open AI model
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=openAIKey
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder
    )

    vector_store.add_documents(split_documents)



