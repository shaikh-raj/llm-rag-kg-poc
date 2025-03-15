import os
import streamlit as st

# LangChain imports
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from langchain_core.documents import Document
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SerpAPIWrapper
from sqlalchemy import create_engine

# Set environment variables and configure Google API
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "SERPAPI_API_KEY" not in os.environ:  # Add SerpAPI key
    os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# LangSmith configuration (optional)
if "LANGSMITH_API_KEY" not in os.environ and "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"

# Neo4j Aura configuration
if "NEO4J_URI" not in os.environ:
    os.environ["NEO4J_URI"] = st.secrets.get("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = st.secrets.get("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = st.secrets.get("NEO4J_PASSWORD")


def create_text_embeddings(text):
    documents = [Document(page_content=text)]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index


def init_knowledge_graph(text):
    documents = [Document(page_content=text)]

    # Create LLM for transformations
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )

    # Initialize Graph
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    # Clear existing graph for demo purposes
    graph.query("MATCH (n) DETACH DELETE n")

    # Transform documents to graph
    llm_transformer = LLMGraphTransformer(llm=gemini_llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Add to graph
    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()

    return graph


def init_database(df):
    # Save DataFrame to SQLite
    conn_str = "sqlite:///./demo_database.db"
    engine = create_engine(conn_str)
    df.to_sql("employees", engine, index=False, if_exists="replace")

    # Create DB connection
    db = SQLDatabase.from_uri(conn_str)
    return db


def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )


def get_qa_chain(retriever):
    llm = get_llm()
    memory = st.session_state.memory

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # explicitly set the output key to "answer"
    )

    return qa_chain


def get_kg_chain():
    llm = get_llm()
    graph = st.session_state.graph

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True
    )

    return chain


def extract_sql_query(text):
    # Look for "SQLQuery:" in the text and extract the query portion
    if "SQLQuery:" in text:
        return text.split("SQLQuery:")[-1].strip()
    return text.strip()


def reframe_sql_output(raw_result):
    # Use the LLM to transform the raw SQL output into a human-friendly answer.
    llm = get_llm()
    prompt = (
        f"The SQL query returned the following result: {raw_result}. "
        "Please reframe this result as a clear and concise answer."
    )
    reframed = llm.invoke(prompt)
    return reframed.content


def get_sql_chain():
    llm = get_llm()
    db = st.session_state.db_connection

    # Create SQL chain using the LLM and the DB.
    sql_chain = create_sql_query_chain(llm, db)

    full_chain = (
        {"question": RunnablePassthrough()}
        | sql_chain
        # Extracts just the SQL part from the output
        | (lambda output: extract_sql_query(output))
        | db.run                                   # Executes the SQL query
        # Uses LLM to reframe the raw SQL output
        | (lambda result: reframe_sql_output(result))
        | StrOutputParser()                        # Parses the output string if necessary
        | (lambda x: {"result": x, "source": "SQL Database"})
    )

    return full_chain


def answer_query(query, use_text, use_db, use_kg, use_web):
    answers = []

    # Text-based RAG
    if use_text and st.session_state.text_vectorstore:
        retriever = st.session_state.text_vectorstore.as_retriever(search_kwargs={
                                                                   "k": 3})
        text_chain = get_qa_chain(retriever)
        text_result = text_chain({"question": query})
        answers.append({
            "result": text_result["answer"],
            "source": "Text RAG",
            "source_docs": text_result.get("source_documents", [])
        })

    # Knowledge Graph
    if use_kg and st.session_state.kg_initialized:
        kg_chain = get_kg_chain()
        kg_result = kg_chain.invoke({"query": query})
        answers.append({
            "result": kg_result["result"],
            "source": "Knowledge Graph",
            "cypher": kg_result.get("cypher", "")
        })

    # Database
    if use_db and st.session_state.db_connection:
        db_chain = get_sql_chain()
        try:
            db_result = db_chain.invoke(query)
            answers.append(db_result)
        except Exception as e:
            answers.append({
                "result": f"Database error: {str(e)}",
                "source": "SQL Database"
            })

    # Web search via SerpAPI with processed output
    if use_web:
        try:
            with st.spinner("Searching the web..."):
                search = SerpAPIWrapper()
                web_result = search.run(query)
                processed_web_result = reframe_web_output(web_result)
                answers.append({
                    "result": processed_web_result,
                    "source": "Web Search"
                })
        except Exception as e:
            answers.append({
                "result": f"Web search error: {str(e)}",
                "source": "Web Search"
            })

    # Combine answers
    if not answers:
        return "Please enable at least one data source and ensure it's properly initialized.", "System Error"

    if len(answers) == 1:
        return answers[0]["result"], answers[0]["source"]
    else:
        # Use LLM to combine multiple answers
        llm = get_llm()

        sources_text = "\n\n".join(
            [f"Source: {a['source']}\nAnswer: {a['result']}" for a in answers])
        prompt = f"""
        I have received multiple answers to the query: "{query}"
        
        {sources_text}
        
        Please synthesize these answers into a comprehensive response. 
        Mention the different sources used when relevant.
        """

        combined_answer = llm.invoke(prompt)
        sources = ", ".join([a["source"] for a in answers])
        return combined_answer.content, f"Combined from: {sources}"


def reframe_web_output(raw_web_result):
    """
    Uses the LLM to process the raw output from the web search into a clear, concise answer.
    """
    llm = get_llm()
    prompt = (
        f"The following is a raw web search result: {raw_web_result}\n\n"
        "Please process and summarize it into a concise and informative answer."
    )
    processed = llm.invoke(prompt)
    return processed.content