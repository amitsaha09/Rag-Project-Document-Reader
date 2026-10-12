from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


file_path = Path(__file__).parent / "document loaders" / "deepLearning.pdf"


docs = PyPDFLoader(
    str(file_path)
).load()

splitter = RecursiveCharacterTextSplitter(
   # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)   


