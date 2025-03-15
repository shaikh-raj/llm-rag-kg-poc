# LLM POC Project

This project demonstrates integration of various AI capabilities using LangChain, including:

- Text-based RAG (Retrieval-Augmented Generation)
- Knowledge Graph integration with Neo4j
- SQL database querying
- Web search capabilities
- Multi-tool orchestration

## Features

- Text input processing with vector embeddings
- Knowledge graph creation and querying
- Database integration for structured data
- Web search capability via MCP (Model Context Protocol)
- Streamlit-based user interface

## Setup Instructions

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up API keys:
   - Create a `secrets.toml` file in the `.streamlit` directory with:
     ```
     GOOGLE_API_KEY = "your-google-api-key"
     LANGSMITH_API_KEY = "your-langsmith-api-key" # Optional
     NEO4J_URI = "your-neo4j-uri"
     NEO4J_USERNAME = "your-neo4j-username"
     NEO4J_PASSWORD = "your-neo4j-password"
     ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Text Data**: Enter text or use the sample text. Click "Process Text Data" to create embeddings and knowledge graph.

2. **Database**: Upload an Excel file or use the sample database. Click "Process Database" to load data into SQLite.

3. **Web Search**: Enable web search for external information retrieval.

4. **Query Selection**: Select which data sources to use for answering queries.

5. **Ask Questions**: Type your query in the text box and get answers from the selected data sources.

## Architecture

The application uses a modular architecture:

- **Streamlit Frontend**: User interface and interaction
- **LangChain**: Orchestration of various components
- **FAISS**: Vector storage for text embeddings
- **Neo4j**: Knowledge graph storage and querying
- **SQLite**: Relational database for structured data
- **MCP**: Model Context Protocol for web search capabilities

## Project Structure

```
├── app.py                 # Main Streamlit application
├── mcp_client.py          # MCP client implementation
├── web_search_server.py   # Web search server implementation
├── requirements.txt       # Project dependencies
├── flow_diagram.png       # System architecture diagram
└── README.md              # Project documentation
```

## Future Improvements

- Support for PDF and document processing
- Integration with more external tools
- Enhanced visualization capabilities
- User authentication and permissions
- Improved performance and caching