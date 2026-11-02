import os

# Set User-Agent before importing LangChain community loaders
os.environ["USER_AGENT"] = "ChatWithWebsite/1.0"

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Application Configuration
st.set_page_config(page_title="Chat with Website (Groq)", page_icon="🌐", layout="wide")

AVAILABLE_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]


def normalize_url(url: str) -> str:
    """Ensures URL has a valid HTTP/HTTPS scheme."""
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        return f"https://{url}"
    return url


def get_retriever_from_url(url: str) -> BM25Retriever:
    """
    Scrapes a website URL and constructs an in-memory BM25 retriever.
    Operates 100% locally with zero external API calls or rate-limit quotas.
    """
    clean_url = normalize_url(url)
    loader = WebBaseLoader(clean_url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    document_chunks = text_splitter.split_documents(documents)

    if not document_chunks:
        raise ValueError("No readable text content could be extracted from the provided URL.")

    retriever = BM25Retriever.from_documents(document_chunks)
    retriever.k = 4
    return retriever


def build_rag_chain(retriever: BM25Retriever, api_key: str, model_name: str = "openai/gpt-oss-20b"):
    """Constructs a history-aware conversational RAG chain powered by Groq."""
    llm = ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0.2,
    )

    # Sub-chain: Reformulate user query using conversation history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Sub-chain: Document question-answering
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI assistant answering questions based strictly on the website context below:\n\n"
            "{context}\n\n"
            "If the answer cannot be found in the context, state clearly that the website does not contain this information. "
            "Keep answers factual, well-structured, and concise.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_retriever, qa_chain)


# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# Sidebar: Configuration & Website Ingestion
with st.sidebar:
    st.title("🌐 Website Setup")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    selected_model = st.selectbox(
        "Open-Source Model",
        options=AVAILABLE_MODELS,
        index=0,
    )

    url_input = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        value=st.session_state.current_url,
    )

    if st.button("Load & Process Website", use_container_width=True):
        if not api_key:
            st.error("Please configure GROQ_API_KEY in .env or enter it above.")
        elif not url_input.strip():
            st.error("Please enter a valid website URL.")
        else:
            with st.spinner("Indexing website content locally..."):
                try:
                    retriever = get_retriever_from_url(url_input)
                    st.session_state.rag_chain = build_rag_chain(retriever, api_key, selected_model)
                    st.session_state.current_url = normalize_url(url_input)
                    st.session_state.chat_history = []
                    st.success("Website successfully indexed!")
                except Exception as e:
                    st.error(f"Failed to process website: {e}")

    if st.session_state.current_url:
        st.divider()
        st.caption(f"**Active Source:**\n`{st.session_state.current_url}`")
        st.caption(f"**Model:** `{selected_model}`")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main Chat View
st.title("💬 Chat with Website")

if not st.session_state.rag_chain:
    st.info("👈 Enter a website URL in the sidebar and click **'Load & Process Website'** to start chatting.")
else:
    # Render message history
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    # Chat Input
    if user_query := st.chat_input("Ask a question about the website..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history,
                    })
                    answer = response.get("answer", "")
                    st.markdown(answer)

                    # Update history
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
