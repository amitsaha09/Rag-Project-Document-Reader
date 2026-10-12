from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="chroma-db", 
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful Ai assistant.
    Use ONLY the provided context to answer the question.
    if the answer is not present in the context, 
    say: "I could not find the answer in the document."""),
    ("human", """Context:{context}
    Question: {question}""")
])

print("\n===== Rag System Created =====\n")

print(" press 0 to exit the program")

while True:
    query = input("\nEnter your question: ")
    if query == "0":
        print("Exiting the program...")
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("\n===== Answer =====\n")
    print(f"AI: {response.content}")