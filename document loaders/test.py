from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

file_path = Path(__file__).parent / "Notes.txt"

ssplitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)
data = TextLoader(
    str(file_path),
    encoding="utf-8"
).load()

chunks = ssplitter.split_documents(data)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content}")
