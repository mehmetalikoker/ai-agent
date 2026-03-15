from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sympy import content

load_dotenv()

aimodel = ChatOpenAI(model="gpt-4")


search = TavilySearchResults(max_results=2)
tavilysearch = [search]

modelEngine = aimodel.bind_tools(tavilysearch)

if __name__ == "__main__":
    response = modelEngine.invoke([HumanMessage(content="what is Mehmet Ali Köker job")])
    print(response)
    print(response.tool_calls)