from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

openai.api_key = os.getenv("OPENAI_API_KEY")
#web search agent
websearch_agent = Agent(
    name  = "WebSearchAgent",
    role  = "Search the web for the information",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True,
)

#financial agent
financial_agent = Agent(
    name = "FinanceAIAgent",
    model = Groq(id="llama-3.1-70b-versatile"),
    tools = [
        YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True),
    ],
    instructions = ["Use table to display the data"],
    show_tool_calls = True,
    markdown = True,
)

multi_ai_agent = Agent(
    team = [websearch_agent,financial_agent],
    model = Groq(id="llama-3.1-70b-versatile"),
    instructions = ["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown = True,
)

multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA",stream=True)

