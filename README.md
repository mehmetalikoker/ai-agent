# What's ai-agent Project
This project features an AI Agent architecture capable of generating responses by leveraging LangChain and LangGraph, with the ability to perform real-time web research via the Tavily infrastructure based on the complexity of the query.
- Langgraph was used in agent management.
- It utilizes Tavily for it's search infrastructure. 
- Additionally, LangSmith was used in the application's monitoring and testing processes.

## Documentation

- [docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/overview) – Comprehensive documentation, including conceptual overviews and guides
- https://github.com/langchain-ai/langchain - For langchain framework usage and details
- https://docs.langchain.com/oss/python/langgraph/workflows-agents - For langgraph agent workflows

## How It Works
Running the class via terminal is sufficient.

For the UI version 
- Terminal -> streamlit run agentwithui.py

For the UI and RAG PDF version 
- Terminal -> streamlit run agentwithrag.py

For the UI and RAG Multiple file version 
- Terminal -> streamlit run agentwithragv2.py

## Requirements
- OPENAI_API_KEY
- LANGCHAIN_API_KEY
- LANGCHAIN_TRACING_V2 Info
- LANGCHAIN_PROJECT Info
- TAVILY_API_KEY

## How Does It Look
<img width="1200" height="560" alt="image" src="https://github.com/user-attachments/assets/291bf8d2-dd4f-4825-94e5-5939e51a66ba" />
