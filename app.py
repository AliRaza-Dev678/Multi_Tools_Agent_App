"""
Multi-tool AI agent (web search, weather, currency conversion, Wikipedia)
served through a Streamlit chat UI, powered by Groq + LangGraph.

Local development:
1. Copy .env.example to .env and fill in your real keys.
2. pip install -r requirements.txt
3. streamlit run app.py

Streamlit Cloud deployment:
1. Push everything EXCEPT .env to GitHub (.gitignore already excludes it).
2. In your Streamlit Cloud app's Settings -> Secrets, paste the contents
   of .streamlit/secrets.toml.example with your real keys filled in.
"""

import json
import os
import time

import requests
import streamlit as st
import wikipedia
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "llama-3.3-70b-versatile"
APP_TITLE = "Multi-Tool AI Agent"

# Groq's llama-3.3-70b-versatile occasionally emits tool calls as raw text
# (e.g. "<function=get_weather_data{...}</function>") instead of a proper
# structured tool_call, which makes the Groq API reject the request with a
# 400 "tool_use_failed" error. This is a known, intermittent provider-side
# issue (not something we can 100% prevent client-side), so we retry.
MAX_TOOL_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.5

SYSTEM_PROMPT = """You are a helpful multi-tool assistant.

You have access to the following tools:
1. search_tool - Search the web for latest news and current events
2. get_weather_data - Fetch real-time weather for any specific city
3. get_conversion_factor - Get currency conversion rate between two currencies
4. convert - Convert an amount using a conversion rate
5. get_wikipedia_summary - Get factual/background summaries on topics, people,
   organizations, acronyms, and sports leagues or tournaments (e.g. "PSL",
   "NASA", "FIFA World Cup")

Before acting, think carefully about what the user is actually asking for:
- A request for background/explanation/"what is X" about an org, acronym,
  league, person, or place -> get_wikipedia_summary
- A request about current conditions outside (temperature, rain, wind) in a
  named city -> get_weather_data
- A request about breaking news or very recent events -> search_tool
- A request to convert money between currencies -> get_conversion_factor
  then convert

If a question requires more than one tool (e.g. "what is the capital of X and
its weather"), call ONE tool at a time and wait for its result before
deciding on the next tool. Do not try to plan or write out multiple tool
calls in a single response.

Do not default to weather or search just because a query is short or has an
acronym. Always use the right tool and explain results in simple friendly
language."""

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="centered")


# ---------------------------------------------------------------------------
# Secrets handling: works both locally (.env) and on Streamlit Cloud (secrets)
# ---------------------------------------------------------------------------
def get_secret(key: str) -> str | None:
    # st.secrets raises if no secrets.toml exists at all (e.g. local dev
    # using only a .env file), so guard the lookup instead of using `in`.
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    load_dotenv()
    return os.getenv(key)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
WEATHERSTACK_API_KEY = get_secret("WEATHERSTACK_API_KEY")
EXCHANGERATE_API_KEY = get_secret("EXCHANGERATE_API_KEY")

missing = [
    name
    for name, val in [
        ("GROQ_API_KEY", GROQ_API_KEY),
        ("WEATHERSTACK_API_KEY", WEATHERSTACK_API_KEY),
        ("EXCHANGERATE_API_KEY", EXCHANGERATE_API_KEY),
    ]
    if not val
]
if missing:
    st.error(
        "Missing required key(s): "
        + ", ".join(missing)
        + ". Add them to a local `.env` file for development, or to your "
        "Streamlit Cloud app's **Secrets** settings for deployment."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def search_tool(query: str) -> str:
    """Search the web for latest news and current information."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current live weather (temperature, wind, humidity, etc.)
    for a specific real-world city name. Only use this when the user is
    asking about current weather conditions, not for general facts.
    """
    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response = requests.get(url, timeout=10)
    return str(response.json())


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> str:
    """Fetches the currency conversion factor between a base and target currency."""
    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}"
        f"/pair/{base_currency}/{target_currency}"
    )
    response = requests.get(url, timeout=10)
    return json.dumps(response.json())


@tool
def convert(base_currency: float, conversion_rate: float) -> float:
    """Given a currency conversion rate, converts a base amount into the target currency."""
    return base_currency * conversion_rate


@tool
def get_wikipedia_summary(query: str) -> str:
    """
    Fetches a short factual background/encyclopedia summary from Wikipedia
    for a given topic, organization, acronym, or proper noun.

    Use this for: countries, history, famous people, science, organizations,
    sports leagues/teams/tournaments (e.g. "PSL", "FIFA World Cup"), and any
    acronym or named entity the user wants general background on.

    Do NOT use this for live data like weather, prices, or scores — use the
    relevant tool for that instead.
    """
    try:
        return wikipedia.summary(query, sentences=4, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Try one of these: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'"


TOOLS = [search_tool, get_weather_data, get_conversion_factor, convert, get_wikipedia_summary]


# ---------------------------------------------------------------------------
# Agent (cached so it's built once per session, not on every message)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        model=MODEL_NAME,
        # temperature=0 makes tool-call generation more deterministic, which
        # reduces (but doesn't fully eliminate) the odds of Groq/Llama
        # emitting a malformed "<function=...>" text blob instead of a
        # proper structured tool call.
        temperature=0,
        max_tokens=1024,
        api_key=GROQ_API_KEY,
    )
    return create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )


agent = get_agent()


# ---------------------------------------------------------------------------
# Helper: invoke the agent with retries for Groq's intermittent
# "tool_use_failed" 400 error (malformed function-call text generation).
# ---------------------------------------------------------------------------
def invoke_agent_with_retry(payload: dict, max_retries: int = MAX_TOOL_RETRIES):
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return agent.invoke(payload)
        except Exception as e:
            last_error = e
            error_text = str(e)

            # Only retry the specific, known-intermittent Groq failure mode.
            # Anything else (auth errors, network errors, etc.) should fail
            # immediately rather than burning retries.
            is_retryable = "tool_use_failed" in error_text or "Failed to call a function" in error_text

            if is_retryable and attempt < max_retries:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue

            raise last_error

    # Should never reach here, but keeps type checkers happy.
    raise last_error


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title(f"🤖 {APP_TITLE}")
st.caption(f"Powered by Groq (`{MODEL_NAME}`) + LangGraph · Web search, weather, currency, Wikipedia")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "**Available tools:**\n"
        "- 🔎 Web search\n"
        "- 🌦️ Weather lookup\n"
        "- 💱 Currency conversion\n"
        "- 📚 Wikipedia summaries"
    )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything — news, weather, currency, facts...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = invoke_agent_with_retry(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
                reply = result["messages"][-1].content
            except Exception as e:
                error_text = str(e)
                if "tool_use_failed" in error_text or "Failed to call a function" in error_text:
                    reply = (
                        "⚠️ The AI model had trouble formatting a tool call for this "
                        "request (a known intermittent issue with this Groq model). "
                        "Please try rephrasing your question, or ask about one thing "
                        "at a time (e.g. first the capital, then the weather)."
                    )
                else:
                    reply = f"⚠️ Something went wrong: {e}"

        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


"""
# Multi-tool AI agent (web search, weather, currency conversion, Wikipedia)
# served through a Streamlit chat UI, powered by Groq + LangGraph.

# Local development:
#     1. Copy .env.example to .env and fill in your real keys.
#     2. pip install -r requirements.txt
#     3. streamlit run app.py

# Streamlit Cloud deployment:
#     1. Push everything EXCEPT .env to GitHub (.gitignore already excludes it).
#     2. In your Streamlit Cloud app's Settings -> Secrets, paste the contents
#        of .streamlit/secrets.toml.example with your real keys filled in.
# """

# import json
# import os

# import requests
# import streamlit as st
# import wikipedia
# from ddgs import DDGS
# from dotenv import load_dotenv
# from langchain_core.messages import SystemMessage
# from langchain_core.tools import tool
# from langchain_groq import ChatGroq
# from langgraph.prebuilt import create_react_agent

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# MODEL_NAME = "llama-3.3-70b-versatile"
# APP_TITLE = "Multi-Tool AI Agent"
# SYSTEM_PROMPT = """You are a helpful multi-tool assistant.
# You have access to the following tools:

# 1. search_tool - Search the web for latest news and current events
# 2. get_weather_data - Fetch real-time weather for any specific city
# 3. get_conversion_factor - Get currency conversion rate between two currencies
# 4. convert - Convert an amount using a conversion rate
# 5. get_wikipedia_summary - Get factual/background summaries on topics, people,
#    organizations, acronyms, and sports leagues or tournaments (e.g. "PSL",
#    "NASA", "FIFA World Cup")

# Before acting, think carefully about what the user is actually asking for:
# - A request for background/explanation/"what is X" about an org, acronym,
#   league, person, or place -> get_wikipedia_summary
# - A request about current conditions outside (temperature, rain, wind) in a
#   named city -> get_weather_data
# - A request about breaking news or very recent events -> searcha_tool
# - A request to convert money between currencies -> get_conversion_factor
#   then convert

# Do not default to weather or search just because a query is short or has an
# acronym. Always use the right tool and explain results in simple friendly
# language."""

# st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="centered")


# # ---------------------------------------------------------------------------
# # Secrets handling: works both locally (.env) and on Streamlit Cloud (secrets)
# # ---------------------------------------------------------------------------
# def get_secret(key: str) -> str | None:
#     # st.secrets raises if no secrets.toml exists at all (e.g. local dev
#     # using only a .env file), so guard the lookup instead of using `in`.
#     try:
#         if key in st.secrets:
#             return st.secrets[key]
#     except Exception:
#         pass
#     load_dotenv()
#     return os.getenv(key)


# GROQ_API_KEY = get_secret("GROQ_API_KEY")
# WEATHERSTACK_API_KEY = get_secret("WEATHERSTACK_API_KEY")
# EXCHANGERATE_API_KEY = get_secret("EXCHANGERATE_API_KEY")

# missing = [
#     name
#     for name, val in [
#         ("GROQ_API_KEY", GROQ_API_KEY),
#         ("WEATHERSTACK_API_KEY", WEATHERSTACK_API_KEY),
#         ("EXCHANGERATE_API_KEY", EXCHANGERATE_API_KEY),
#     ]
#     if not val
# ]
# if missing:
#     st.error(
#         "Missing required key(s): "
#         + ", ".join(missing)
#         + ". Add them to a local `.env` file for development, or to your "
#         "Streamlit Cloud app's **Secrets** settings for deployment."
#     )
#     st.stop()


# # ---------------------------------------------------------------------------
# # Tools
# # ---------------------------------------------------------------------------
# @tool
# def search_tool(query: str) -> str:
#     """Search the web for latest news and current information."""
#     with DDGS() as ddgs:
#         results = list(ddgs.text(query, max_results=3))
#         return "\n".join(f"{r['title']}: {r['body']}" for r in results)


# @tool
# def get_weather_data(city: str) -> str:
#     """
#     Fetches the current live weather (temperature, wind, humidity, etc.)
#     for a specific real-world city name. Only use this when the user is
#     asking about current weather conditions, not for general facts.
#     """
#     url = (
#         "https://api.weatherstack.com/current"
#         f"?access_key={WEATHERSTACK_API_KEY}&query={city}"
#     )
#     response = requests.get(url, timeout=10)
#     return str(response.json())


# @tool
# def get_conversion_factor(base_currency: str, target_currency: str) -> str:
#     """Fetches the currency conversion factor between a base and target currency."""
#     url = (
#         f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}"
#         f"/pair/{base_currency}/{target_currency}"
#     )
#     response = requests.get(url, timeout=10)
#     return json.dumps(response.json())


# @tool
# def convert(base_currency: float, conversion_rate: float) -> float:
#     """Given a currency conversion rate, converts a base amount into the target currency."""
#     return base_currency * conversion_rate


# @tool
# def get_wikipedia_summary(query: str) -> str:
#     """
#     Fetches a short factual background/encyclopedia summary from Wikipedia
#     for a given topic, organization, acronym, or proper noun.
#     Use this for: countries, history, famous people, science, organizations,
#     sports leagues/teams/tournaments (e.g. "PSL", "FIFA World Cup"), and any
#     acronym or named entity the user wants general background on.
#     Do NOT use this for live data like weather, prices, or scores — use the
#     relevant tool for that instead.
#     """
#     try:
#         return wikipedia.summary(query, sentences=4, auto_suggest=True)
#     except wikipedia.exceptions.DisambiguationError as e:
#         return f"Ambiguous query. Try one of these: {e.options[:5]}"
#     except wikipedia.exceptions.PageError:
#         return f"No Wikipedia page found for '{query}'"


# TOOLS = [search_tool, get_weather_data, get_conversion_factor, convert, get_wikipedia_summary]


# # ---------------------------------------------------------------------------
# # Agent (cached so it's built once per session, not on every message)
# # ---------------------------------------------------------------------------
# @st.cache_resource
# def get_agent():
#     llm = ChatGroq(
#         model=MODEL_NAME,
#         temperature=0.2,
#         max_tokens=1024,
#         api_key=GROQ_API_KEY,
#     )
#     return create_react_agent(
#         model=llm,
#         tools=TOOLS,
#         prompt=SystemMessage(content=SYSTEM_PROMPT),
#     )


# agent = get_agent()


# # ---------------------------------------------------------------------------
# # Chat state
# # ---------------------------------------------------------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []


# # ---------------------------------------------------------------------------
# # UI
# # ---------------------------------------------------------------------------
# st.title(f"🤖 {APP_TITLE}")
# st.caption(f"Powered by Groq (`{MODEL_NAME}`) + LangGraph · Web search, weather, currency, Wikipedia")

# with st.sidebar:
#     st.header("Settings")
#     st.markdown(
#         "**Available tools:**\n"
#         "- 🔎 Web search\n"
#         "- 🌦️ Weather lookup\n"
#         "- 💱 Currency conversion\n"
#         "- 📚 Wikipedia summaries"
#     )
#     if st.button("🗑️ Clear chat", use_container_width=True):
#         st.session_state.messages = []
#         st.rerun()

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# user_input = st.chat_input("Ask me anything — news, weather, currency, facts...")

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             try:
#                 result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
#                 reply = result["messages"][-1].content
#             except Exception as e:
#                 reply = f"⚠️ Something went wrong: {e}"
#         st.markdown(reply)

#     st.session_state.messages.append({"role": "assistant", "content": reply})
