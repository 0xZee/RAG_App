import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_groq import ChatGroq
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
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
from streamlit_pdf_viewer import pdf_viewer


# Set page configuration
st.set_page_config(page_title="RAG_Bot", layout="wide")

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
        #self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        self.embeddings = CohereEmbeddings(model="embed-english-v3.0")

        self.llm = ChatGroq(
            model = "meta-llama/llama-4-scout-17b-16e-instruct", # "llama3-8b-8192",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.2
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

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
        original_filename = uploaded_file.name.replace(".", "_")  # Use original filename for collection name

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load and split the document
            loader = self.get_document_loader(tmp_file_path, file_type)
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)

            # Create a new collection for this upload using original filename
            collection_name = f"{original_filename}_{datetime.now().strftime('%Y%m%d%H%M')}"
            # Replace any invalid characters for MongoDB collection names
            collection_name = collection_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
            collection = self.client[self.db_name][collection_name]

            # Create vectorstore
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                relevance_score_fn="cosine"
            )

            # Add documents to VectorStore with metadata if PDF
            if file_type == "pdf":
                # Create PDF metadata
                pdf_id = str(uuid.uuid4())[:8]
                metadatas = []

                for i, doc in enumerate(texts):
                    meta = {
                        "source_page": doc.metadata.get("page", i + 1),
                        "title": f"Document {pdf_id}",
                        "chunk_count": len(texts)
                    }
                    metadatas.append(meta)

                ids = [f"id_{i}" for i in range(len(texts))]
                vector_store.add_documents(documents=texts, ids=ids, metadatas=metadatas)
            else:
                ids = [f"id_{i}" for i in range(len(texts))]
                vector_store.add_documents(documents=texts, ids=ids)

            # Now create index search after documents are added
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": 1024,
                            "path": "embedding",
                            "similarity": "cosine"
                        },
                        {
                          "type": "filter",
                          "path": "source_page"
                        }
                    ]
                },
                name=f"{collection_name}_index",
                type="vectorSearch"
            )

            result = collection.create_search_index(model=search_index_model)
            print("New search index named " + result + " is building.")

            # Update the vector store with the index name
            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name=f"{collection_name}_index",
                relevance_score_fn="cosine"
            )

            return vector_store, collection_name

        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)

    def setup_retriever(self, vectorstore):
        """Set up a retriever from the vectorstore with Cohere reranking"""
        # Set up base retriever
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Retrieve more documents initially for reranking
        )

        # Set up Cohere reranker
        compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=4)

        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
            #search_kwargs={"k": 6}  # Return top 6 after reranking
        )

        return compression_retriever

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
            compression_retriever = self.setup_retriever(vector_store)
            retrieved_docs = compression_retriever.invoke(question)
            #retrieved_docs = vector_store.similarity_search(question)
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
        st.subheader("📚 RAG Bot - :orange-background[Bases de Connaissances] ", divider='orange')

        # Initialize RAG engine
        rag_engine = RAGEngine()

        # File upload component
        uploaded_file = st.file_uploader(
            "Upload a document", 
            type=["pdf", "txt", "csv", "docx"]
        )

        # Process button
        if uploaded_file:
            # Display PDF viewer if PDF
            if uploaded_file.name.endswith(".pdf"):
                # Save the uploaded PDF to a temporary file for viewing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_pdf_path = tmp_file.name
                # Display the PDF
                pdf_viewer(tmp_pdf_path, height=380)

            if st.button("♻️ Index Document", use_container_width=True, type='primary'):
                with st.spinner("❇️ Indexation du Document..."):
                    # Process uploaded document
                    vectorstore, collection_name = rag_engine.setup_vectorstore(uploaded_file)

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

                    st.success(f"✅ Document traité : {uploaded_file.name}")
                    st.info(f"📋 Base de Connaissance créée : {collection_name}")

        # Clear chat button
        if st.session_state.conversation is not None:
            if st.button("Re-Initialiser le Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()

# Chat interface
def chat_interface():
    st.subheader("RAG_Bot - Chat with your Uploaded Documents", divider='orange')

    # Add greeting message if this is the first time showing the chat
    if len(st.session_state.messages) == 0:
        greeting = "Bonjour! Je suis votre assistant RAG. Posez-moi des questions sur le document que vous venez de télécharger!"
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
    if prompt := st.chat_input("Posez une question sur votre document..."):
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
                        st.info(doc.page_content)
                        st.write("----------------")

# Main function
def main():
    sidebar()

    # Display welcome message or chat interface based on state
    if st.session_state.conversation is None:
        st.subheader("📚 :orange[RAG_Bot] - Upload & Chat with Documents", divider='orange')
        st.info("""
        👋 Bienvenue sur RAG_Bot!

        Pour commencer:
        1. Téléchargez un document depuis la barre latérale (PDF, TXT, CSV, DOCX).
        2. Cliquez sur le bouton "Process Document" pour traiter le document.
        3. Une fois traité, vous pourrez poser des questions sur votre document.

        🧩 Ce chatbot utilise MongoDB Atlas Vector Search, Cohere embeddings, et Groq LLM pour fournir des réponses pertinentes à partir de vos documents.
        """)
    else:
        chat_interface()

if __name__ == "__main__":
    main()