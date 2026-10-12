from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

file_path = Path(__file__).parent / "RUG.pdf"

data = PyPDFLoader(
    str(file_path)
).load()

splits = splitter.split_documents(data)

print(splits[0].page_content)