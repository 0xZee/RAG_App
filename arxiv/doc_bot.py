import streamlit as st
import os
from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_groq import ChatGroq
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from pymongo import MongoClient
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

# DocEngine class to handle existing collections and conversation
class DocEngine:
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

    def get_available_collections(self):
        """Get list of collections in the IT_OPS database"""
        try:
            collections = self.client[self.db_name].list_collection_names()
            return collections
        except Exception as e:
            st.error(f"Error fetching collections: {e}")
            return []

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
        """Set up a retriever from the vectorstore with Cohere reranking"""
        # Set up base retriever
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 14}  # Retrieve more documents initially for reranking
        )

        # Set up Cohere reranker
        compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)

        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
            #search_kwargs={"k": 9}  # Return top 6 after reranking
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
        Context: \n {context}
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
            docs_content = "\n---\n".join(doc.page_content for doc in state["context"])
            formatted_prompt = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(formatted_prompt)
            return {"answer": response.content}

        # Compile application
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph

# Sidebar UI Components
def sidebar():
    with st.sidebar:
        st.subheader("📚 :orange[Doc_Bot] - Explorer les :orange-background[Bases de Connaissances]", divider="grey")

        # Initialize Doc engine
        doc_engine = DocEngine()

        # Get available collections
        collections = doc_engine.get_available_collections()

        # Data source selection
        collection_name = st.selectbox(
            "📂 Choisissez une Base de Connaissance :",
            ["None"] + collections,
            index=0
        )

        # Process selection button
        if collection_name != "None":
            #if st.button("Se connecter à la collection", use_container_width=True):
            with st.spinner("❇️ Connexion à la collection..."):
                # Connect to existing collection
                vectorstore = doc_engine.get_vectorstore(collection_name)

                # Set up retriever and conversation
                retriever = doc_engine.setup_retriever(vectorstore)
                conversation = doc_engine.setup_conversation(vectorstore)

                # Add memory for checkpointing
                memory = MemorySaver()

                # Update session state
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.conversation = conversation
                st.session_state.memory = memory

                st.success(f"✅ Connecté à la base de connaissance : *{collection_name}*")

        # Clear chat button
        if st.session_state.conversation is not None:
            if st.button("✨ Re-Initialiser le Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()

# Chat interface
def chat_interface():
    st.subheader("Doc_Bot - Chat with your Document Collections", divider='orange')

    # Add greeting message if this is the first time showing the chat
    if len(st.session_state.messages) == 0:
        greeting = "Bonjour! Je suis votre assistant documentaire. Posez-moi des questions sur la collection sélectionnée!"
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
    if prompt := st.chat_input("Posez une question sur la collection..."):
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
                        st.info(f'*{doc.page_content}*')
                        st.write("----------------")

# Main function
def main():
    sidebar()

    # Display welcome message or chat interface based on state
    if st.session_state.conversation is None:
        st.subheader("🔍 :orange[Doc_Bot] - Explorer vos Collections", divider='orange')
        st.info("""
        👋 Bienvenue sur Doc_Bot!

        Pour commencer:
        1. Sélectionnez une collection existante dans la base de données IT_OPS depuis la barre latérale.
        2. Cliquez sur le bouton "Se connecter à la collection" pour établir la connexion.
        3. Une fois connecté, vous pourrez poser des questions sur les documents de cette collection.

        Ce chatbot utilise MongoDB Atlas Vector Search, Cohere embeddings, et Groq LLM pour fournir des réponses pertinentes à partir de vos collections de documents.

        Pour télécharger de nouveaux documents et créer de nouvelles collections, utilisez l'application RAG_Bot.
        """)
    else:
        chat_interface()

if __name__ == "__main__":
    main()