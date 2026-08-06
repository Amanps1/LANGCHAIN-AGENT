import os
import certifi
from dotenv import load_dotenv
import streamlit as st
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain import hub
from langchain.tools import tool
import requests
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["WEATHERSTACK_API_KEY"] = WEATHERSTACK_API_KEY

search_tool = TavilySearchResults(
    max_results=2
)

@tool
def get_weather_data(city: str) -> str:
    """
    Get current weather information for a specified city.

    Args:
        city: Name of the city to get weather details.

    Returns:
        Temperature, weather condition, and humidity information.
    """

    url = (
        "https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(url)
        data = response.json()
        if "current" not in data:
            return f"Could not fetch weather data for {city}. Error: {data}"

        current = data["current"]

        return (
            f"Weather Report for {city}\n"
            f"Temperature: {current['temperature']}°C\n"
            f"Condition: {current['weather_descriptions'][0]}\n"
            f"Humidity: {current['humidity']}%"
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"
    
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)

prompt = hub.pull(
    "hwchase17/react"
)

tools=[search_tool, get_weather_data]


agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)


if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")