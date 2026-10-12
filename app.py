import streamlit as st
from pathlib import Path
import shutil
import tempfile

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 Book RAG Assistant")
st.write(
    "Upload a PDF book and ask questions about its content."
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_DIR = "chroma-db"


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar - Upload Book
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload Your Book")

    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"]
    )

    create_database = st.button(
        "Create Knowledge Base",
        use_container_width=True
    )

    if uploaded_file is not None:

        st.info(
            f"Selected book: **{uploaded_file.name}**"
        )


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

if create_database:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF book first."
        )

    else:

        with st.spinner(
            "Processing your book... This may take a while."
        ):

            try:

                # ------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = temp_file.name


                # ------------------------------------------
                # Load PDF
                # ------------------------------------------

                loader = PyPDFLoader(
                    temp_pdf_path
                )

                documents = loader.load()


                # ------------------------------------------
                # Split documents into chunks
                # ------------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    documents
                )


                # ------------------------------------------
                # Create embeddings
                # ------------------------------------------

                embedding_model = OpenAIEmbeddings()


                # ------------------------------------------
                # Remove old Chroma database
                # ------------------------------------------

                chroma_path = Path(CHROMA_DIR)

                if chroma_path.exists():

                    shutil.rmtree(chroma_path)


                # ------------------------------------------
                # Create new Chroma database
                # ------------------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=CHROMA_DIR
                )


                # ------------------------------------------
                # Save in session state
                # ------------------------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.book_name = (
                    uploaded_file.name
                )

                st.session_state.messages = []


                st.success(
                    f"Knowledge base created successfully!"
                )

                st.write(
                    f"📄 **Pages:** {len(documents)}"
                )

                st.write(
                    f"🧩 **Chunks:** {len(chunks)}"
                )


            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

st.divider()

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF book from the sidebar and "
        "click **Create Knowledge Base** to get started."
    )

else:

    st.subheader(
        f"💬 Ask questions about: "
        f"{st.session_state.book_name}"
    )


    # --------------------------------------------------
    # Display previous messages
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------
    # Chat input
    # --------------------------------------------------

    query = st.chat_input(
        "Ask a question about the book..."
    )


    if query:

        # ----------------------------------------------
        # Display user question
        # ----------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):

            st.markdown(query)


        # ----------------------------------------------
        # Retrieve relevant documents
        # ----------------------------------------------

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
        )


        docs = retriever.invoke(query)


        # ----------------------------------------------
        # Create context
        # ----------------------------------------------

        context = "\n\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )


        # ----------------------------------------------
        # Prompt
        # ----------------------------------------------

        prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
                You are a helpful AI assistant.

                Answer the question using ONLY the
                provided context.

                If the answer is not present in the
                context, say:

                "I could not find the answer in the document."
                """
            ),

            (
                "human",
                """
                Context:

                {context}

                Question:

                {question}
                """
            )
        ])


        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })


        # ----------------------------------------------
        # LLM
        # ----------------------------------------------

        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )


        # ----------------------------------------------
        # Generate answer
        # ----------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = llm.invoke(
                    final_prompt
                )

                answer = response.content

                st.markdown(answer)


        # ----------------------------------------------
        # Save assistant response
        # ----------------------------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


        # ----------------------------------------------
        # Show sources
        # ----------------------------------------------

        with st.expander(
            "📚 View Retrieved Sources"
        ):

            for i, doc in enumerate(docs):

                page_number = (
                    doc.metadata.get(
                        "page",
                        "Unknown"
                    )
                )

                st.markdown(
                    f"**Source {i + 1} — "
                    f"Page {page_number + 1 if isinstance(page_number, int) else page_number}**"
                )

                st.write(
                    doc.page_content[:1000]
                )

                st.divider()