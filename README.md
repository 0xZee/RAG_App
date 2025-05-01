# Document RAG Assistant Suite

A suite of Streamlit applications that enable conversational interactions with document collections using Retrieval-Augmented Generation (RAG).

## 🌟 Overview

This project consists of two complementary Streamlit applications:

1. **RAG_Bot** - Upload and process new documents to create searchable knowledge bases.
2. **Doc_Bot** - Connect to existing document collections and query them conversationally.

Both applications utilize MongoDB Atlas Vector Search for document storage and retrieval, Cohere for embeddings, and Groq LLM for generating high-quality responses in French.

## 🚀 Features

### RAG_Bot
- Upload various document types (PDF, TXT, CSV, DOCX)
- Automatic document processing and chunking
- Vector embedding generation with Cohere
- MongoDB Atlas Vector Search integration
- Real-time chat interface with the processed document
- Source citation and reference

### Doc_Bot
- Connect to existing document collections in MongoDB
- Browse available knowledge bases
- Chat-based querying of document collections
- Source citation and reference
- Seamless integration with collections created by RAG_Bot

## ⚙️ Technical Stack

- **Frontend**: Streamlit
- **Vector Database**: MongoDB Atlas Vector Search
- **Embeddings**: Cohere (embed-multilingual-v3.0)
- **LLM**: Groq (meta-llama/llama-4-scout-17b-16e-instruct)
- **RAG Framework**: LangChain + LangGraph
- **Document Processing**: LangChain document loaders

## 📋 Requirements

- Python 3.9+
- Streamlit
- MongoDB Atlas account
- Cohere API key
- Groq API key

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/doc-rag-assistant.git
cd doc-rag-assistant
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Set up your `.streamlit/secrets.toml` file with your API keys:

```toml
MDB_URI_CONNECTION = "mongodb+srv://username:password@your-cluster.mongodb.net/?retryWrites=true&w=majority"
COHERE_API_KEY = "your-cohere-api-key"
GROQ_API_KEY = "your-groq-api-key"
```

## 🚀 Usage

### Running RAG_Bot

```bash
streamlit run rag_bot.py
```

1. Upload a document using the file uploader in the sidebar
2. Click "Process Document" to create a new collection in MongoDB
3. Chat with your document using the chat interface
4. Note the collection name for future reference with Doc_Bot

### Running Doc_Bot

```bash
streamlit run doc_bot.py
```

1. Select an existing collection from the dropdown in the sidebar
2. Click "Se connecter à la collection" to establish the connection
3. Chat with the selected document collection using the chat interface

## 🌐 Workflow

The typical workflow involves:

1. Using RAG_Bot to upload and process new documents, creating collections in MongoDB
2. Using Doc_Bot to access and query those collections later

This separation allows for a clean division between document processing and querying functionality.

## 🔍 How It Works

1. **Document Processing**:
   - Documents are loaded using LangChain document loaders
   - Text is split into manageable chunks with RecursiveCharacterTextSplitter
   - Chunks are embedded using Cohere's multilingual embedding model
   - Embeddings are stored in MongoDB Atlas with vector search capabilities

2. **Retrieval Process**:
   - User questions are embedded using the same Cohere model
   - MongoDB Atlas Vector Search finds semantically similar document chunks
   - Retrieved chunks provide context for the LLM

3. **Response Generation**:
   - LangGraph orchestrates the retrieval and generation flow
   - Groq LLM (LLama 4) generates responses based on retrieved context
   - Responses are provided in French with proper source citations

## 📝 Notes

- All responses are generated in French as specified in the prompt template
- Document collections are stored in the "IT_OPS" database in MongoDB Atlas
- Collections created by RAG_Bot are prefixed with "doc_" and include a timestamp
- Source references are provided with each response

## ⚠️ Requirements File

Create a `requirements.txt` file with the following dependencies:

```
streamlit
pymongo
langchain
langchain-core
langchain-mongodb
langchain-cohere
langchain-groq
langchain-community
langgraph
python-dotenv
```

## 🛠️ Configuration

Ensure your MongoDB Atlas cluster has:
1. Vector search capability enabled
2. Proper network access settings
3. A user with read/write permissions

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
