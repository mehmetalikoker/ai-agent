from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

aimodel = ChatOpenAI(model="gpt-4")
memory = SqliteSaver.from_conn_string("memory.text")
search = TavilySearchResults(max_results=3)

tavilysearch = [search]

modelEngine = aimodel.bind_tools(tavilysearch)

agent = create_react_agent(aimodel,tavilysearch,checkpointer=memory)

config = {"configurable":{"thread_id":"abc1"}}

if __name__ == "__main__":
    while True:
        user_input = input("> ")
        for chunk in agent.stream(
                { "messages": [HumanMessage(content=user_input)]},
            config=config
        ):
            print(chunk)
            print("----")
