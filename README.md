# 🤖 Multi-Tool AI Agent App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct%20Agent-6B4FBB?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-F55036?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A conversational AI agent built with **Groq**, **LangGraph**, and **Streamlit** that autonomously selects and uses the right tool to answer any user query — from live weather and real-time news to currency conversion and Wikipedia facts.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Agent Tools](#-agent-tools)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Keys Setup](#-api-keys-setup)
- [Deployment](#-deployment-on-streamlit-cloud)
- [How It Works](#-how-it-works)
- [Example Queries](#-example-queries)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

This project implements a **multi-tool ReAct agent** — an AI that doesn't just chat, but reasons about *which tool to use* and *when* to best answer a question. Powered by Meta's **LLaMA 3.3 70B** model via Groq's ultra-fast inference API, the agent dynamically routes queries to the appropriate tool and returns clear, friendly responses through a Streamlit chat interface.

---

## 🌐 Live Demo

> Deploy your own instance on [Streamlit Cloud](https://streamlit.io/cloud) — see the [Deployment](#-deployment-on-streamlit-cloud) section below.

---

## ✨ Features

- 💬 **Conversational chat UI** built with Streamlit
- 🧠 **Autonomous tool selection** — the agent decides which tool to use based on the query
- ⚡ **Ultra-fast inference** via Groq API (LLaMA 3.3 70B)
- 🔄 **ReAct architecture** via LangGraph — reason, act, observe, repeat
- 🔐 **Dual secrets management** — works with `.env` locally and Streamlit Cloud secrets in production
- 🗑️ **Session memory** with a one-click chat reset

---

## 🛠 Agent Tools

The agent has access to **4 specialized tools** it selects from autonomously:

| Tool | Description | API Used |
|---|---|---|
| 🔎 **Web Search** | Searches the web for latest news and current events | DuckDuckGo Search (DDGS) |
| 🌦️ **Weather Lookup** | Fetches real-time weather for any city (temperature, wind, humidity) | Weatherstack API |
| 💱 **Currency Conversion** | Gets live exchange rates and converts amounts between currencies | ExchangeRate API |
| 📚 **Wikipedia Summary** | Retrieves factual background on topics, people, organizations, and acronyms | Wikipedia Python Library |

> The agent uses a carefully crafted system prompt to prevent tool misuse — for example, it won't call the weather tool for a general factual question about a city, and won't call Wikipedia for a live data request.

---

## 🧰 Tech Stack

| Technology | Role |
|---|---|
| **Python 3.8+** | Core language |
| **Streamlit** | Chat UI and web app framework |
| **LangGraph** | ReAct agent orchestration (`create_react_agent`) |
| **LangChain (Core + Groq)** | Tool definitions and LLM integration |
| **Groq API** | Ultra-fast LLM inference (LLaMA 3.3 70B Versatile) |
| **DuckDuckGo Search (ddgs)** | No-auth web search |
| **Weatherstack API** | Real-time weather data |
| **ExchangeRate API** | Live currency conversion rates |
| **Wikipedia** | Encyclopedic background summaries |
| **python-dotenv** | Local environment variable management |

---

## 📁 Project Structure

```
Multi_Tools_Agent_App/
│
├── app.py                  # Main Streamlit app — agent, tools, and UI
├── agent.ipynb             # Jupyter notebook — prototyping and experimentation
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes .env and secrets from version control
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- API keys for Groq, Weatherstack, and ExchangeRate (see [API Keys Setup](#-api-keys-setup))

### 1. Clone the Repository

```bash
git clone https://github.com/AliRaza-Dev678/Multi_Tools_Agent_App.git
cd Multi_Tools_Agent_App
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
WEATHERSTACK_API_KEY=your_weatherstack_api_key_here
EXCHANGERATE_API_KEY=your_exchangerate_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔑 API Keys Setup

You need **3 free API keys**:

| Service | Where to Get It | Free Tier |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | ✅ Free |
| **Weatherstack** | [weatherstack.com](https://weatherstack.com) | ✅ Free (1,000 calls/month) |
| **ExchangeRate API** | [exchangerate-api.com](https://www.exchangerate-api.com) | ✅ Free (1,500 calls/month) |

> **Note:** DuckDuckGo Search (`ddgs`) requires no API key.

---

## ☁️ Deployment on Streamlit Cloud

1. Push your code to GitHub (**do not include `.env`** — it is already in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository
3. In **Settings → Secrets**, add your keys in TOML format:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
WEATHERSTACK_API_KEY = "your_weatherstack_api_key_here"
EXCHANGERATE_API_KEY = "your_exchangerate_api_key_here"
```

4. Click **Deploy** — the app handles both local `.env` and Streamlit Cloud secrets automatically.

---

## ⚙️ How It Works

```
User Message
     │
     ▼
LangGraph ReAct Agent (LLaMA 3.3 70B via Groq)
     │
     ├─── Reason: What does the user need?
     │
     ├─── Act: Select the right tool
     │         ├── 🔎 search_tool
     │         ├── 🌦️ get_weather_data
     │         ├── 💱 get_conversion_factor + convert
     │         └── 📚 get_wikipedia_summary
     │
     ├─── Observe: Process tool output
     │
     └─── Respond: Generate a clear, friendly answer
```

The **ReAct (Reason + Act)** pattern allows the agent to chain multiple tool calls when needed — for example, first fetching a conversion rate and then applying it to the requested amount.

---

## 💬 Example Queries

| Query | Tool Used |
|---|---|
| `"What's the weather in Karachi right now?"` | 🌦️ Weather |
| `"Convert 500 USD to PKR"` | 💱 Currency Conversion |
| `"What is the PSL?"` | 📚 Wikipedia |
| `"Latest news about AI today"` | 🔎 Web Search |
| `"Who is Elon Musk?"` | 📚 Wikipedia |
| `"What's happening in Pakistan today?"` | 🔎 Web Search |
| `"How much is 1 EUR in JPY?"` | 💱 Currency Conversion |
| `"What is NASA?"` | 📚 Wikipedia |

---

## 🔮 Future Improvements

- [ ] Add **conversation memory** so the agent recalls earlier messages in a session
- [ ] Integrate a **calculator tool** for general math queries
- [ ] Add a **time & timezone tool** for world clock queries
- [ ] Support **multi-language responses** via LLM language detection
- [ ] Add **tool usage indicators** in the UI to show which tool was called
- [ ] Write **unit tests** for each tool function

---

## 👤 Author

**Ali Raza**
- GitHub: [@AliRaza-Dev678](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
