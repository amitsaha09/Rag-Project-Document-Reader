from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI Book"}),
    Document(page_content="Pandas is used for data manipulation and analysis in Python.", metadata={"source": "Data Science Book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "Deep Learning Book"})
]

embedding_model = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embedding_model,
    persist_directory="chroma-db"
)

#retrievers
result = vectorstore.similarity_search("What is use for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Explain deep learning")

for doc in docs:
    
    print(doc.page_content)
    print(doc.metadata)