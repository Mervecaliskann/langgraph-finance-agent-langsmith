"""
AI Finance Agent (Real-Time Stock Analysis)
-------------------------------------------
Author: Merve Çalışkan
Role: AI Engineer & Data Scientist
Date: 2026-02-10

Description:
This application is an autonomous AI Agent capable of retrieving real-time 
stock market data, analyzing financial metrics, and providing investment 
insights using Llama-3 via Groq API.

Tech Stack:
- Orchestration: LangChain (Tool Calling Agent)
- LLM: Llama-3-70b-versatile (Groq)
- Data Source: Yahoo Finance (yfinance)
- UI: Streamlit

Key Features:
1. Real-time Price Tracking
2. Fundamental Data Analysis (P/E Ratio, Market Cap)
3. Technical Summary Generation
"""

import streamlit as st
import yfinance as yf
import os
from dotenv import load_dotenv
from datetime import datetime

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. KONFIGÜRASYON VE AYARLAR
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="📈 AI Financial Analyst", layout="wide")

# API Key Kontrolü (Güvenlik)
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ HATA: .env dosyasında GROQ_API_KEY bulunamadı!")
    st.stop()

# 2. ARAÇLARIN TANIMLANMASI (TOOLS)
# Ajanın "elleri ve gözleri". LLM bu fonksiyonları ne zaman çalıştıracağına kendi karar verir.
# ---------------------------------------------------------

@tool
def get_stock_price(ticker: str):
    """
    Retrieves the current stock price and daily change percentage.
    Input example: 'AAPL', 'TSLA', 'THYAO.IS' (for Turkish stocks).
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if history.empty:
            return {"error": "Veri bulunamadı. Sembolü kontrol et."}
        
        current_price = history['Close'].iloc[-1]
        open_price = history['Open'].iloc[-1]
        change_percent = ((current_price - open_price) / open_price) * 100
        
        return {
            "symbol": ticker.upper(),
            "price": f"{current_price:.2f}",
            "change_percent": f"%{change_percent:.2f}"
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_company_fundamentals(ticker: str):
    """
    Retrieves fundamental financial data like Market Cap, P/E Ratio, and Sector.
    Use this when the user asks for 'analysis', 'details', or 'company info'.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Company": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "Summary": info.get("longBusinessSummary", "")[:300] + "..." # Özetin başı
        }
    except Exception as e:
        return {"error": str(e)}

# Araçları bir listeye koyuyoruz
tools = [get_stock_price, get_company_fundamentals]

# 3. AI AJANININ KURULUMU (BRAIN)
# ---------------------------------------------------------
def initialize_agent():
    # Model: Llama-3-70b (Hızlı ve zeki)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0, # Finansal veride halüsinasyon istemiyoruz, 0 olmalı.
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Prompt: Ajana kim olduğunu öğretiyoruz
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Senior Financial Analyst AI. "
            "Your goal is to provide accurate market data and insights. "
            "Use the provided tools to fetch real-time data from Yahoo Finance. "
            "Always be professional, concise, and data-driven. "
            "If you calculate something, explain your logic. "
            "Do NOT make up numbers. Only use data from tools."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # Ajanın düşünme alanı
    ])

    # Ajanı ve Çalıştırıcıyı (Executor) oluştur
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. ARAYÜZ (STREAMLIT UI)
# ---------------------------------------------------------
st.title("📈 AI Financial Analyst Agent")
st.markdown("---")

# Yan Panel (Sidebar)
with st.sidebar:
    st.header("🔍 Kullanım Kılavuzu")
    st.markdown("""
    Bu ajan **Yahoo Finance** verilerini canlı olarak analiz eder.
    
    **Örnek Sorular:**
    - *Apple (AAPL) hissesi bugün ne kadar?*
    - *Türk Hava Yolları (THYAO.IS) temel analizi nedir?*
    - *Microsoft ve Google'ı karşılaştır.*
    """)
    st.info("Developed by Merve Çalışkan using LangChain & Groq")

# Sohbet Geçmişi (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan Girdi Alma
if user_input := st.chat_input("Hisse senedi hakkında soru sor..."):
    # 1. Kullanıcı mesajını ekrana bas
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Ajanın düşünmesi ve cevap vermesi
    with st.chat_message("assistant"):
        agent_executor = initialize_agent()
        
        with st.spinner("Piyasa verileri analiz ediliyor..."):
            try:
                # Ajanı çalıştır
                response = agent_executor.invoke({"input": user_input})
                output_text = response["output"]
                
                st.markdown(output_text)
                
                # Cevabı hafızaya kaydet
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")