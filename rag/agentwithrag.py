import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import os

load_dotenv()
st.set_page_config(page_title="PDF Architect Agent", layout="wide")


# PDF Process and Prepare RAG---
def process_pdf(uploaded_file):
    # Temporary save for PDF file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # PDF load and file split
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Creating Vector DB
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Tool creating for Agent
    # create_retriever_tool --> The agent now chooses when to check the web (Tavily) and when to look at the PDF you uploaded
    pdf_tool = create_retriever_tool(
        retriever,
        "pdf_search",
        "Use this tool to search for information within the uploaded document."
    )
    return pdf_tool


# Load Agent
st.title("📄 PDF & Web Research Agent")


# Sidebar: PDF Load Area
with st.sidebar:
    st.header("Upload document")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")
    if uploaded_file:
        st.success("PDF downloaded successfully !")
        if "pdf_tool" not in st.session_state:
            with st.spinner("The document is being analyzed..."):
                st.session_state.pdf_tool = process_pdf(uploaded_file)



@st.cache_resource
def get_base_tools():
    return [TavilySearchResults(max_results=3)]


# Agent create with dynamic tools
# If you haven't uploaded the PDF, the agent will only perform an internet search.
tools = get_base_tools()
if "pdf_tool" in st.session_state:
    tools.append(st.session_state.pdf_tool)


memory = SqliteSaver.from_conn_string("memory.db")
aimodel = ChatOpenAI(model="gpt-4-turbo", streaming=True)
agent = create_react_agent(aimodel, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "pdf_session_1"}}


# UI and Chat
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("PDF hakkında veya internetten bir şey sor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        with st.status("Düşünülüyor...", expanded=True) as status:
            full_response = ""
            for chunk in agent.stream({"messages": [HumanMessage(content=prompt)]}, config=config):
                for node, values in chunk.items():
                    st.write(f"🔄 `{node}` aşaması çalışıyor...")
                    if "messages" in values:
                        last_msg = values["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            full_response = last_msg.content
            status.update(label="Yanıt Hazır!", state="complete", expanded=False)

        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})