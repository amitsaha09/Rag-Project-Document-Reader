from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
   # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
file_path = Path(__file__).parent / "RUG.pdf"

data = PyPDFLoader(
    str(file_path)
).load()

splits = splitter.split_documents(data)

print(splits[0].page_content)