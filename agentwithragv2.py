import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration & Page Setup ---
load_dotenv()
st.set_page_config(page_title="Universal Knowledge Agent", layout="wide")


# --- 1. Multi-File Processing Logic ---
def process_multiple_files(uploaded_files):
    all_docs = []
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Automatically detects file type (PDF, Docx, TXT, etc.)
        loader = UnstructuredFileLoader(file_path)
        all_docs.extend(loader.load())

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Vector Store Creation
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return create_retriever_tool(
        retriever,
        "multi_doc_search",
        "Use this tool to search through the uploaded documents (PDF, Word, Excel, Text)."
    )


# --- 2. Sidebar & File Uploads ---
st.title("📂 Universal Knowledge Agent")

with st.sidebar:
    st.header("Document Repository")
    uploaded_files = st.file_uploader(
        "Drop files here (PDF, DOCX, TXT, CSV...)",
        accept_multiple_files=True
    )

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Indexing all documents..."):
            st.session_state.doc_tool = process_multiple_files(uploaded_files)
            st.success(f"{len(uploaded_files)} files are ready!")


# --- 3. Agent Initialization ---
@st.cache_resource
def get_base_tools():
    return [TavilySearchResults(max_results=3)]


tools = get_base_tools()
if "doc_tool" in st.session_state:
    tools.append(st.session_state.doc_tool)

# Persisting memory in SQLite
memory = SqliteSaver.from_conn_string("memory.db")
llm = ChatOpenAI(model="gpt-4-turbo", streaming=True)

# System Prompt to define the agent's persona
system_message = "You are a highly skilled Software Architect. Provide technical, concise, and accurate information."

agent = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    prompt=system_message
)
# --- 4. Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about documents or the web..."):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Processing data...", expanded=False) as status:
            full_response = ""
            config = {"configurable": {"thread_id": "multi_doc_session_en"}}

            # Stream the agent's response steps
            for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                for key, value in chunk.items():
                    st.write(f"Executing step: `{key}`")
                    if "messages" in value:
                        msg = value["messages"][-1]
                        if isinstance(msg, AIMessage) and msg.content:
                            full_response = msg.content

            status.update(label="Response Ready!", state="complete")

        # Display final response
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})