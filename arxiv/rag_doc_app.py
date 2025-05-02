import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import tempfile
import uuid
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate

# Set page configuration
st.set_page_config(page_title="Doc_Bot", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Memory saver class for LangGraph checkpointing
class MemorySaver:
    def __init__(self):
        self.memory = {}

    def get(self, key: str) -> dict:
        return self.memory.get(key, None)

    def put(self, key: str, value: dict) -> None:
        self.memory[key] = value

# RAGEngine class to handle document processing and conversation
class RAGEngine:
    def __init__(self):
        # Initialize MongoDB client
        self.mongodb_uri = st.secrets["MDB_URI_CONNECTION"]
        self.client = MongoClient(self.mongodb_uri)
        self.db_name = "IT_OPS"

        # Initialize embeddings and LLM
        api_key = st.secrets["COHERE_API_KEY"]
        os.environ["COHERE_API_KEY"] = api_key
        self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

        self.llm = ChatGroq(
            model = "meta-llama/llama-4-scout-17b-16e-instruct", # "llama3-8b-8192",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.2
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def get_available_collections(self):
        """Get list of collections in the IT_OPS database"""
        try:
            collections = self.client[self.db_name].list_collection_names()
            return collections
        except Exception as e:
            st.error(f"Error fetching collections: {e}")
            return []

    def get_document_loader(self, file_path, file_type):
        """Returns the appropriate document loader based on file type"""
        if file_type == "pdf":
            return PyPDFLoader(file_path)
        elif file_type == "txt":
            return TextLoader(file_path)
        elif file_type == "csv":
            return CSVLoader(file_path)
        elif file_type == "docx":
            return Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def setup_vectorstore(self, uploaded_file):
        """Process an uploaded file and create a vectorstore"""
        file_type = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and split the document
            loader = self.get_document_loader(tmp_file_path, file_type)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)

            # Create a new collection for this upload
            collection_name = f"doc_{datetime.now().strftime('%Y%m%d%H%M')}"
            collection = self.client[self.db_name][collection_name]

            # create index search
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": 1024,
                            "path": "embedding",
                            "similarity": "cosine"
                        }
                    ]
                },
                name=f"{collection_name}_index",
                type="vectorSearch"
            )
            result = collection.create_search_index(model=search_index_model)
            print("New search index named " + result + " is building.")

            # Create vectorstore
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name=f"{collection_name}_index",
                relevance_score_fn="cosine"
            )

            # Add documents to vectorstore
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            vector_store.add_documents(documents=texts, ids=ids)

            # Create vector search index (dimensions for Cohere embed-multilingual-v3.0)
            vector_store.create_vector_search_index(dimensions=1024)

            return vector_store

        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)

    def get_vectorstore(self, collection_name):
        """Get an existing vectorstore from MongoDB Atlas"""
        collection = self.client[self.db_name][collection_name]

        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name=f"{collection_name}_index",
            relevance_score_fn="cosine"
        )

        return vector_store

    def setup_retriever(self, vectorstore):
        """Set up a retriever from the vectorstore"""
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

    def setup_conversation(self, vector_store):
        """Set up a LangGraph conversation chain with the vector store"""

        # Define the RAG prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are an french assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know.
        Réponds toujours en français, avec une réponse détaillé et claire, en mentionnant les sources en format markdown.
        ---
        Context: {context} \n\n
        ---
        Question: {question}

        Answer:
        """)

        # Define state for application
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
            sources: List[Document]

        # Define application steps
        def retrieve(state: State) -> Dict[str, Any]:
            """Retrieve relevant documents from vector store"""
            question = state["question"]
            retrieved_docs = vector_store.similarity_search(question)
            return {"context": retrieved_docs, "sources": retrieved_docs}

        def generate(state: State) -> Dict[str, Any]:
            """Generate answer based on retrieved context"""
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            formatted_prompt = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(formatted_prompt)
            return {"answer": response.content}

        # Compile application
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        # Compile the graph
        graph = graph_builder.compile()

        return graph

# UI Components
def sidebar():
    with st.sidebar:
        st.subheader("RAG Chatbot Settings")

        # Initialize RAG engine
        rag_engine = RAGEngine()

        # Get available collections
        collections = rag_engine.get_available_collections()

        # Data source selection
        st.subheader("Select Knowledge Base")

        # Create options list with None and Upload options
        options = ["None"] + collections + ["Upload New Document"]

        # Select data source with None as first option
        data_source = st.selectbox(
            "Choose a collection or upload a file:",
            options
        )

        # Handle file upload option
        uploaded_file = None
        if data_source == "Upload New Document":
            uploaded_file = st.file_uploader(
                "Upload a document", 
                type=["pdf", "txt", "csv", "docx"]
            )

        # Process selection button
        if data_source != "None":
            if (data_source in collections or 
                (data_source == "Upload New Document" and uploaded_file)):

                #if st.button("Process Selection"):
                with st.spinner("❇️ Indexation du Doc/Contexte..."):
                    # Process based on selection
                    if data_source in collections:
                        # Connect to existing collection
                        vectorstore = rag_engine.get_vectorstore(data_source)
                        success_msg = f"✅ Connecté : {data_source} "
                    else:
                        # Process uploaded document
                        vectorstore = rag_engine.setup_vectorstore(uploaded_file)
                        success_msg = f"Document processed: {uploaded_file.name}"

                    # Set up retriever and conversation
                    retriever = rag_engine.setup_retriever(vectorstore)
                    conversation = rag_engine.setup_conversation(vectorstore)

                    # Add memory for checkpointing
                    memory = MemorySaver()

                    # Update session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.conversation = conversation
                    st.session_state.memory = memory

                    st.success(success_msg)

        # Clear chat button
        if st.session_state.conversation is not None:
            if st.button("Initialiser le Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()

# Chat interface
def chat_interface():
    st.subheader("Doc_Bot - Chat with your Docs", divider='orange')

    # Add greeting message if this is the first time showing the chat
    if len(st.session_state.messages) == 0:
        greeting = "Hello! I'm your RAG assistant. Ask me anything about your selected documents!"
        st.session_state.messages.append(AIMessage(content=greeting))

    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get response from conversation chain
        with st.chat_message("assistant"):
            with st.spinner("❇️ Génération de réponse..."):
                response = st.session_state.conversation.invoke({
                    "question": prompt
                })

                answer = response["answer"]
                st.write(answer)

                # Update session state
                st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.chat_history.append((prompt, answer))

                # Display sources in expander
                with st.expander("Sources"):
                    for i, doc in enumerate(response["sources"]):
                        st.write(f"Source {i+1}:")
                        st.write(doc.page_content)
                        st.write("----------------")

# Main function
def main():
    sidebar()

    # Display welcome message or chat interface based on state
    if st.session_state.conversation is None:
        st.subheader("📚 :orange[:material/DOC_BOT:] - Chat with Docs", divider='orange')
        st.info("""
        👋 Welcome to the Doc_Bot - Chat with Docs!

        To get started:
        1. Select a knowledge base from the sidebar dropdown (collections from IT_OPS database).
        2. If uploading a document, select the file from your computer.
        3. Click the "Process Selection" button to set up the chat.
        4. Once connected, you can ask questions about your documents.

        This chatbot uses MongoDB Atlas Vector Search, Cohere embeddings, and Groq LLM to provide relevant answers from your documents.
        """)
    else:
        chat_interface()

if __name__ == "__main__":
    main()