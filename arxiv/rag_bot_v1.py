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

# --- Page setup ---
st.set_page_config(page_title="RAG_Bot", layout="wide")

# --- Session state vars ---
if "messages"      not in st.session_state: st.session_state.messages = []
if "chat_history"  not in st.session_state: st.session_state.chat_history = []
if "vectorstore"   not in st.session_state: st.session_state.vectorstore = None
if "retriever"     not in st.session_state: st.session_state.retriever = None
if "chat_engine"   not in st.session_state: st.session_state.chat_engine = None
if "memory"        not in st.session_state: st.session_state.memory = None

# --- Memory saver for LangGraph checkpointing ---
class MemorySaver:
    def __init__(self):
        self.memory = {}
    def get(self, key: str) -> dict:
        return self.memory.get(key, None)
    def put(self, key: str, value: dict) -> None:
        self.memory[key] = value

# --- RAGEngine class ---
class RAGEngine:
    def __init__(self):
        # MongoDB
        self.mongodb_uri = st.secrets["MDB_URI_CONNECTION"]
        self.client = MongoClient(self.mongodb_uri)
        self.db_name = "IT_OPS"
        # Embeddings & LLM
        os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
        self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0.2
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def get_document_loader(self, file_path, file_type):
        if file_type == "pdf": return PyPDFLoader(file_path)
        if file_type == "txt": return TextLoader(file_path)
        if file_type == "csv": return CSVLoader(file_path)
        if file_type == "docx": return Docx2txtLoader(file_path)
        raise ValueError(f"Unsupported file type: {file_type}")

    def setup_vectorstore(self, uploaded_file):
        file_type = uploaded_file.name.split(".")[-1].lower()
        original_filename = uploaded_file.name.replace(".", "_")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        try:
            loader = self.get_document_loader(tmp_path, file_type)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)

            # collection name: original + timestamp
            coll_name = f"{original_filename}_{datetime.now():%Y%m%d%H%M}".replace("/", "_")
            coll = self.client[self.db_name][coll_name]

            vs = MongoDBAtlasVectorSearch(collection=coll, embedding=self.embeddings, relevance_score_fn="cosine")
            ids = [f"id_{i}" for i in range(len(splits))]
            if file_type == "pdf":
                pdf_id = str(uuid.uuid4())[:8]
                metas = [{"source_page": d.metadata.get("page", i+1),
                          "title": f"Document {pdf_id}", "chunk_count": len(splits)}
                         for i, d in enumerate(splits)]
                vs.add_documents(documents=splits, ids=ids, metadatas=metas)
            else:
                vs.add_documents(documents=splits, ids=ids)

            # build search index
            idx = SearchIndexModel(
                definition={"fields":[
                    {"type":"vector","numDimensions":1024,"path":"embedding","similarity":"cosine"},
                    {"type":"filter","path":"source_page"}
                ]},
                name=f"{coll_name}_index", type="vectorSearch"
            )
            coll.create_search_index(model=idx)
            # re-init vs with index name
            vs = MongoDBAtlasVectorSearch(
                collection=coll, embedding=self.embeddings,
                index_name=f"{coll_name}_index", relevance_score_fn="cosine"
            )
            return vs, coll_name
        finally:
            os.unlink(tmp_path)

    def setup_retriever(self, vectorstore):
        base = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})
        reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=4)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)

    def setup_conversation(self, vector_store):
        prompt = ChatPromptTemplate.from_template("""
You êtes un assistant français pour questions-réponses.
Utilisez ces contextes pour répondre. Si inconnu, dites-le.
Répondez toujours en français, détaillé et clair, en mentionnant les sources en markdown.
---
Context: {context}

Question: {question}

Réponse:
""")
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
            sources: List[Document]

        def retrieve(state: State):
            retr = self.setup_retriever(vector_store)
            docs = retr.invoke(state["question"])
            return {"context": docs, "sources": docs}

        def generate(state: State):
            txt = "\n\n".join(d.page_content for d in state["context"])
            full = prompt.invoke({"question": state["question"], "context": txt})
            res = self.llm.invoke(full)
            return {"answer": res.content}

        gb = StateGraph(State).add_sequence([retrieve, generate])
        gb.add_edge(START, "retrieve")
        return gb.compile()

    def setup_simple_conversation(self):
        prompt = ChatPromptTemplate.from_template("""
Tu es un assistant IA. Réponds clairement et concisément en français.

Question: {question}

Réponse:
""")
        class State(TypedDict):
            question: str
            answer: str

        def gen(state: State):
            full = prompt.invoke({"question": state["question"]})
            res = self.llm.invoke(full)
            return {"answer": res.content}

        gb = StateGraph(State).add_node("generate", gen)
        gb.add_edge(START, "generate")
        return gb.compile()

# --- Sidebar UI ---
def sidebar():
    with st.sidebar:
        st.subheader("📚 RAG Bot - Bases de Connaissances", divider='orange')
        rag_engine = RAGEngine()

        use_context = st.toggle("🔍 Chat avec Document", value=False)

        # Nouveau Chat button
        if st.button("💬 Nouveau Chat", use_container_width=True):
            # reset history
            st.session_state.messages = []
            st.session_state.chat_history = []

            if not use_context:
                st.session_state.chat_engine = rag_engine.setup_simple_conversation()
            else:
                if st.session_state.vectorstore is None:
                    st.error("⚠️ Aucun document indexé. Chargez-en un d’abord.")
                else:
                    st.session_state.chat_engine = rag_engine.setup_conversation(
                        st.session_state.vectorstore
                    )
            st.rerun()

        # Only if context mode is ON do we show uploader + indexer
        if use_context:
            st.markdown("---")
            uploaded_file = st.file_uploader(
                "📄 Ajouter votre document", type=["pdf", "txt", "csv", "docx"]
            )
            if uploaded_file:
                if uploaded_file.name.endswith(".pdf"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    pdf_viewer(tmp_path, height=380)
                if st.button("♻️ Indexer le document", use_container_width=True, type="primary"):
                    with st.spinner("Indexation en cours…"):
                        vs, coll_name = rag_engine.setup_vectorstore(uploaded_file)
                        retr = rag_engine.setup_retriever(vs)
                        conv = rag_engine.setup_conversation(vs)
                        mem  = MemorySaver()
                        st.session_state.vectorstore  = vs
                        st.session_state.retriever    = retr
                        st.session_state.chat_engine  = conv
                        st.session_state.memory       = mem
                        st.success(f"✅ Document traité : {uploaded_file.name}")
                        st.info(f"📋 Base créée : {coll_name}")

        # Clear chat button if a session exists
        #if st.session_state.chat_engine is not None:
        #    if st.button("🔄 Re‑initialiser le Chat", use_container_width=True):
        #        st.session_state.messages     = []
        #        st.session_state.chat_history = []
        #        st.rerun()

# --- Chat Interface ---
def chat_interface():
    st.subheader("Chat", divider="orange")
    if not st.session_state.messages:
        st.session_state.messages.append(
            AIMessage(content="Bonjour ! Comment puis‑je vous aider ?")
        )

    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

    if prompt := st.chat_input("Posez votre question…"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Génération…"):
                out = st.session_state.chat_engine.invoke({"question": prompt})
                ans = out["answer"]
                st.write(ans)
                st.session_state.messages.append(AIMessage(content=ans))
                st.session_state.chat_history.append((prompt, ans))

                # If RAG mode, show sources
                if "sources" in out:
                    with st.expander("Sources"):
                        for i, d in enumerate(out["sources"]):
                            st.write(f"Source {i+1}:")
                            st.info(d.page_content)
                            st.write("---")

# --- Main ---
def main():
    sidebar()
    if st.session_state.chat_engine is None:
        st.subheader("📚 RAG_Bot - Upload & Chat", divider="orange")
        st.info("""
👋 Bienvenue !  
1. Activez “Utiliser le contexte” pour la RAG ou laissez-le désactivé pour une conversation simple.  
2. Si RAG, chargez et indexez votre document.  
3. Cliquez sur “Nouveau Chat” et posez vos questions.
""")
    else:
        chat_interface()

if __name__ == "__main__":
    main()
