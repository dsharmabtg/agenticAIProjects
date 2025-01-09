from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
import openai
import os
from dotenv import load_dotenv
import phi
from phi.playground import Playground, serve_playground_app

#load the environment variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(agents=[financial_agent,websearch_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)
