import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
st.set_page_config(page_title="AI Architect Agent", layout="centered")
st.title("🏗️ Online Agent")

@st.cache_resource
def init_agent():

    memory = SqliteSaver.from_conn_string("memory.db")
    aimodel = ChatOpenAI(model="gpt-4", streaming=True)
    search = TavilySearchResults(max_results=3)
    # Create Agent
    return create_react_agent(aimodel, [search], checkpointer=memory)


agent = init_agent()
# Save Correspondence History with thread_id
config = {"configurable": {"thread_id": "architect_session_1"}}

# Streamlit Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI bind for old chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Mesajınızı yazın..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Agent Answer
    with st.chat_message("assistant"):
        response_container = st.empty()
        final_text = ""

        # LangGraph Stream Process
        for chunk in agent.stream(
                {"messages": [HumanMessage(content=user_query)]},
                config=config
        ):
            for key, value in chunk.items():
                if "messages" in value:
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        final_text = last_msg.content
                        response_container.markdown(final_text)

        # Agent Answer Save
        st.session_state.chat_history.append({"role": "assistant", "content": final_text})