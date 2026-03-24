import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by agent"""

    url: str = Field(description="THe URL of the source")


class AgentResponse(BaseModel):
    """Schema for a response from the agent"""

    answer: str = Field(description="The response from the agent")
    sources: List[Source] = Field(
        default_factory=list, description="The sources used to answer the question"
    )


load_dotenv()

'''
tavily = TavilyClient()

#@tool
#def calculate(number: int) -> int:
    """
    Tool that calculates the factorial of a number
    Args:
        number: the number to calculate the factorial of
    Returns:
        the factorial of the number
    """
    print(f"Calculating the factorial of {number}")
    result = 1
    for i in range(1, number + 1):
        result *= i
    return result

@tool
def search(query: str) -> str:
    """
    Tool that searches the web
    Args:
        query: the query to search for
    Returns:
        the search results
    """
    print(f"Searching for {query}")
    result = tavily.search(query=query)
    return str(result)
'''

llm = ChatOllama(temperature=0, model="gpt-oss:20b")
# tools = [calculate, search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    print(os.getenv("TAVILY_API_KEY"))
    # result = agent.invoke({"messages":[HumanMessage(content="What is the factorial of average tempeature in RDU, in centigrade? ")]})
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the average tempeature in RDU?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
