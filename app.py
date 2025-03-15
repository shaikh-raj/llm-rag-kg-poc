import os
import pandas as pd
from PIL import Image
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import *

# LangSmith configuration (optional)
if "LANGSMITH_API_KEY" not in os.environ and "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"

# Default sample text content
DEFAULT_TEXT = None
with open('default_data.txt', 'r') as f:
    DEFAULT_TEXT = f.read()

# Default sample Excel data
@st.cache_data
def get_default_excel_data():
    data = {
        "model_id": list(range(1, 11)),
        "model_name": [
            "GPT-4", "PaLM 2", "LLaMA 2", "Gemini", "ChatGPT", 
            "Bard", "Claude", "Gopher", "Megatron-Turing", "Jurassic-2"
        ],
        "release_date": [
            "2023-03-14", "2023-05-10", "2023-07-18", "2023-12-01", "2023-11-30", 
            "2023-09-01", "2023-10-15", "2023-08-20", "2023-06-05", "2023-04-20"
        ],
        "num_parameters": [
            "175B", "540B", "70B", "100B", "175B", 
            "100B", "52B", "280B", "530B", "178B"
        ],
        "capabilities": [
            "Advanced reasoning, multimodal support",
            "Multilingual, complex problem-solving, coding",
            "Efficient, open source, fine-tunable",
            "Multimodal input, real-time search integration",
            "Conversational, broad knowledge base",
            "Real-time web search, interactive dialogue",
            "Contextual understanding, dynamic dialogue",
            "Scientific research, reasoning",
            "Large-scale text generation, fine-tuning",
            "Creative generation, contextual reasoning"
        ],
        "architecture": [
            "Transformer", "Transformer", "Transformer", "Transformer", "Transformer", 
            "Transformer", "Transformer", "Transformer", "Transformer", "Transformer"
        ],
        "training_data": [
            "WebText, Books, Code", 
            "Web, Books, Code, Scientific papers", 
            "Diverse text corpus", 
            "Multimodal datasets, Web data", 
            "Conversational data, Web", 
            "Search-indexed web data", 
            "Dialogue datasets, Books", 
            "Books, Articles, Web", 
            "Web, Books, Scientific articles", 
            "Web, News, Social media"
        ],
        "usage": [
            "Chatbots, Content Generation",
            "Enterprise solutions, Content creation",
            "Research, Custom AI applications",
            "Digital assistants, Enterprise tools",
            "Conversational AI, Customer support",
            "Search integration, Interactive dialogue",
            "Customer support, Virtual assistants",
            "Academic research, Advanced reasoning",
            "Large-scale generation, NLP research",
            "Creative applications, Content innovation"
        ]
    }
    return pd.DataFrame(data)


# Initialize session state
def init_session_state():
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "text_vectorstore" not in st.session_state:
        st.session_state.text_vectorstore = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "kg_initialized" not in st.session_state:
        st.session_state.kg_initialized = False
    if "db_connection" not in st.session_state:
        st.session_state.db_connection = None
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )


# Main application (UI part)
def main():
    # Inject CSS styles for containers and scrollable conversation history
    st.markdown("""
    <style>
    .chat-history {
        height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        # background-color: #fff;
    }
    .container-header {
        background-color: #f0f0f0;
        padding: 1px;
        border-radius: 5px;
    }
    .container-body {
        background-color: #f9f9f9;
        padding: 1px;
        border-radius: 5px;
    }
    .container-sidebar {
        background-color: #e8f4f8;
        padding: 1px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()

    # Header Container
    with st.container():
        # st.markdown('<div class="container-header">', unsafe_allow_html=True)
        st.title("LLM POC Project")
        st.write("This tool integrates RAG, Knowledge Graph, SQL & Web Search to generate dynamic responses from text or Excel or Internet data.")

        st.subheader("Tools & Techniques")
        st.markdown("""
        - **Gemini:** LLM Embedding & Text Generation  
        - **LangSmith:** LLM Logging  
        - **LangChain:** Chaining  
        - **FAISS:** VectorDB for RAG  
        - **Streamlit:** FrontEnd  
        - **SQLite:** Database  
        - **Neo4j Aura:** Knowledge Graph  
        - **SerpAPI:** Web Search  
        """)
        st.markdown('</div class="container-sidebar">', unsafe_allow_html=True)

    # Sidebar for data input
    with st.sidebar:
        st.header("Data Sources")

        # Text Configuration
        st.subheader("Text Data")
        use_default_text = st.checkbox("Use sample text", value=True)
        if use_default_text:
            input_text = st.text_area(
                "Text Content", value=DEFAULT_TEXT, height=200, disabled=True)
        else:
            input_text = st.text_area("Text Content", height=200)
        process_text = st.button("Process Text Data")
        if process_text and input_text:
            with st.spinner("Processing text data..."):
                # Create vector store
                st.session_state.text_vectorstore = create_text_embeddings(
                    input_text)
                # Initialize knowledge graph
                try:
                    st.session_state.graph = init_knowledge_graph(input_text)
                    st.session_state.kg_initialized = True
                    st.success("✅ Text processed and Knowledge Graph created!")
                except Exception as e:
                    st.error(f"Error initializing knowledge graph: {str(e)}")

        # Database Configuration
        st.subheader("Database")
        use_default_db = st.checkbox("Use sample database", value=True)
        if use_default_db:
            df = get_default_excel_data()
            st.dataframe(df, height=150)
        else:
            uploaded_file = st.file_uploader(
                "Upload Excel file", type=['xlsx', 'csv'])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.dataframe(df, height=150)
            else:
                df = None
        process_db = st.button("Process Database")
        if process_db and df is not None:
            with st.spinner("Processing database..."):
                try:
                    st.session_state.db_connection = init_database(df)
                    st.success("✅ Database initialized!")
                except Exception as e:
                    st.error(f"Error initializing database: {str(e)}")

        # Web Search Configuration
        # st.subheader("Web Search")
        # use_web_search = st.checkbox("Enable Web Search", value=False)
        # if use_web_search and not os.getenv("SERPAPI_API_KEY"):
        #     st.warning("Please set SERPAPI_API_KEY in secrets to enable web search")

    # Main Chat Interface Container
    with st.container():
        st.markdown('<div class="container-body">', unsafe_allow_html=True)
        st.header("Search Mode")

        # Query selection checkboxes
        col1, col2 = st.columns(2)
        with col1:
            use_text = st.checkbox("Use Text", value=True)
        with col2:
            use_db = st.checkbox("Use Database", value=False)
        col3, col4 = st.columns(2)
        with col3:
            use_kg = st.checkbox("Use Knowledge Graph", value=False)
        with col4:
            use_web = st.checkbox("Use Web Search", value=False)

        # Validate selections
        if use_text and st.session_state.text_vectorstore is None:
            st.warning("Please process text data first from left panel to use Text RAG.")
        if use_kg and not st.session_state.kg_initialized:
            st.warning("Please process text data first from left panel to use Knowledge Graph.")
        if use_db and st.session_state.db_connection is None:
            st.warning("Please process database first from left panel to use SQL queries.")
        if use_web and not os.getenv("SERPAPI_API_KEY"):
            st.warning("Web Search cannot be used.")

        # Chat input
        query = st.text_input("Ask a question:")
        if query:
            with st.spinner("Thinking..."):
                try:
                    answer, source = answer_query(
                        query, use_text, use_db, use_kg, use_web)
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": query})
                    st.session_state.conversation_history.append(
                        {"role": "assistant", "content": answer, "source": source})
                    st.subheader("Chatbot Response")
                    st.write(answer)
                    st.caption(f"Source: {source}")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Conversation History Container (fixed scrollable)
    st.subheader("Conversation History")
    history_html = '<div class="chat-history">'
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            history_html += f'<p><strong>👤 You:</strong> {message["content"]}</p>'
        else:
            history_html += f'<p><strong>🤖 Assistant:</strong> {message["content"]}</p>'
            if "source" in message:
                history_html += f'<p style="font-size: small; color: gray;">Source: {message["source"]}</p>'
            history_html += '<hr>'
    history_html += '</div>'
    st.markdown(history_html, unsafe_allow_html=True)

    # System Architecture Container
    with st.container():
        st.markdown('<div class="container-body">', unsafe_allow_html=True)
        st.subheader("System Architecture")
        try:
            image = Image.open('mermaid-diagram.png')
            st.image(image)
        except:
            st.info("Flow diagram not available.")


if __name__ == "__main__":
    main()
