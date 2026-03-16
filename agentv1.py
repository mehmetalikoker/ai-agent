from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sympy import content
from langgraph.prebuilt import create_react_agent

load_dotenv()

aimodel = ChatOpenAI(model="gpt-4")

search = TavilySearchResults(max_results=6)
tavilysearch = [search]

modelEngine = aimodel.bind_tools(tavilysearch)

agent = create_react_agent(aimodel,tavilysearch)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [HumanMessage(content="Who is software architect martin fowler")]},
    )
    for message in response["messages"]:
        print(message.content)