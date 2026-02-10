# 📈 AI Financial Analyst Agent

A real-time financial analysis agent powered by **Llama-3.3 (Groq)** and **Yahoo Finance**. This agent uses "Tool Calling" capabilities to fetch live stock data, analyze fundamentals, and provide investment insights in natural language.

## 🚀 Features

* **Real-Time Data:** Fetches live stock prices and daily changes using `yfinance`.
* **Fundamental Analysis:** Retrieves key metrics like Market Cap, P/E Ratio, and Sector information.
* **Agentic Workflow:** The AI autonomously decides which tool to use based on the user's question (e.g., price lookup vs. company info).
* **High Speed:** Powered by Groq's LPU inference engine for sub-second responses.

## 🛠️ Tech Stack

* **Brain:** Llama-3.3-70b-versatile (via Groq API)
* **Orchestration:** LangChain (Tool Calling Agent)
* **Data Source:** Yahoo Finance API
* **UI:** Streamlit
* **Language:** Python 3.10+

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mervecaliskann/AI-Finance-Agent.git](https://github.com/Mervecaliskann/AI-Finance-Agent.git)
    cd AI-Finance-Agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Usage

1.  Enter a stock ticker or company name (e.g., "Apple", "THYAO.IS", "Tesla").
2.  Ask questions like:
    * *"What is the current price of Apple?"*
    * *"Give me a fundamental analysis of Microsoft."*
    * *"Compare the PE ratio of Google and Meta."*

---
*Developed by Merve Çalışkan*