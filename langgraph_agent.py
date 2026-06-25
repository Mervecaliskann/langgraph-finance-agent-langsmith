"""
AI Finance Agent - LangGraph Version
-------------------------------------
This module re-implements the tool-calling agent from app.py as an explicit
LangGraph state machine. It is standalone (does not import app.py) and has
no Streamlit dependency, so it can be used from scripts (e.g. eval.py).

Graph:
    fetch_data -> analyze -> [conditional]
        risk_level == "high" -> human_review -> generate_report -> END
        risk_level == "low"  -> generate_report -> END
"""

import os
from typing import TypedDict

import yfinance as yf
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------
PE_RATIO_HIGH_THRESHOLD = 30        # P/E above this -> "high" risk
DAILY_CHANGE_HIGH_THRESHOLD = 3.0   # |daily change %| above this -> "high" risk


# ---------------------------------------------------------
# 1. TOOLS (same logic as app.py)
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
            return {"error": "Veri bulunamadi. Sembolu kontrol et."}

        current_price = history["Close"].iloc[-1]
        open_price = history["Open"].iloc[-1]
        change_percent = ((current_price - open_price) / open_price) * 100

        return {
            "symbol": ticker.upper(),
            "price": f"{current_price:.2f}",
            "change_percent": f"%{change_percent:.2f}",
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
            "Summary": info.get("longBusinessSummary", "")[:300] + "...",
        }
    except Exception as e:
        return {"error": str(e)}


TOOLS = [get_stock_price, get_company_fundamentals]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------
# 2. STATE
# ---------------------------------------------------------
class AgentState(TypedDict):
    input: str
    tools_called: list
    tool_results: dict
    risk_level: str
    warnings: list
    report: str


# ---------------------------------------------------------
# 3. LLM
# ---------------------------------------------------------
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


FETCH_SYSTEM_PROMPT = """You are a financial data retrieval assistant.
Identify which company/companies the user is asking about and call the
appropriate tools to fetch real data for them.

Rules:
- For questions about price, current value, daily change, or volatility, call `get_stock_price`.
- For questions about fundamentals, analysis, market cap, P/E ratio, sector, 52-week high, or company info, call `get_company_fundamentals`.
- If the user asks for a "full analysis" or both price and fundamentals, call BOTH tools.
- If multiple companies are mentioned (e.g. comparisons), call the relevant tool(s) once per company.
- Convert company names to their stock ticker symbols (e.g. Apple -> AAPL, Google -> GOOGL, Meta -> META, Tesla -> TSLA, Microsoft -> MSFT, Amazon -> AMZN, Nvidia -> NVDA, Netflix -> NFLX, Coca-Cola -> KO).
- Always call at least one tool. Do not answer directly without calling a tool.
- When you need data for multiple companies, call tools one at a time sequentially, never call multiple tools in parallel in a single message.
"""

REPORT_SYSTEM_PROMPT = """You are a Senior Financial Analyst AI.
Answer the user's question using ONLY the data provided below. Do NOT make
up numbers. Be professional, concise, and data-driven. If a risk warning is
present, include it prominently at the start of your response.
"""


# ---------------------------------------------------------
# 4. NODES
# ---------------------------------------------------------
def fetch_data(state: AgentState) -> dict:
    """Ask the LLM which tool(s) to call for the user's question, then run them."""
    llm_with_tools = get_llm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=FETCH_SYSTEM_PROMPT),
        HumanMessage(content=state["input"]),
    ]
    ai_msg = llm_with_tools.invoke(messages)

    tools_called = []
    tool_results = {}

    for tool_call in ai_msg.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        ticker = args.get("ticker", "UNKNOWN")

        tool_fn = TOOLS_BY_NAME.get(name)
        if tool_fn is None:
            continue

        result = tool_fn.invoke(args)
        tools_called.append(name)
        tool_results.setdefault(name, {})[ticker] = result

    return {"tools_called": tools_called, "tool_results": tool_results}


def analyze(state: AgentState) -> dict:
    """Inspect the fetched data and decide a risk_level ('high' or 'low')."""
    risk_level = "low"

    fundamentals = state.get("tool_results", {}).get("get_company_fundamentals", {})
    for data in fundamentals.values():
        pe_ratio = data.get("P/E Ratio")
        if isinstance(pe_ratio, (int, float)) and pe_ratio > PE_RATIO_HIGH_THRESHOLD:
            risk_level = "high"

    prices = state.get("tool_results", {}).get("get_stock_price", {})
    for data in prices.values():
        change_str = data.get("change_percent", "")
        try:
            change_value = float(str(change_str).replace("%", ""))
            if abs(change_value) > DAILY_CHANGE_HIGH_THRESHOLD:
                risk_level = "high"
        except (ValueError, TypeError):
            pass

    return {"risk_level": risk_level}


def human_review(state: AgentState) -> dict:
    """Placeholder for a human-in-the-loop step: just attaches a warning."""
    warning = (
        "HIGH RISK DETECTED: This analysis indicates elevated risk "
        "(high P/E ratio and/or high daily volatility). A human reviewer "
        "should validate this before any investment decision."
    )
    return {"warnings": state.get("warnings", []) + [warning]}


def generate_report(state: AgentState) -> dict:
    """Use the LLM to produce the final natural-language report."""
    llm = get_llm()

    data_lines = []
    for tool_name, by_ticker in state.get("tool_results", {}).items():
        for ticker, data in by_ticker.items():
            data_lines.append(f"[{tool_name}] {ticker}: {data}")
    data_block = "\n".join(data_lines) if data_lines else "No data retrieved."

    context_parts = [
        f"User question: {state['input']}",
        f"Data gathered:\n{data_block}",
        f"Risk level: {state.get('risk_level', 'low')}",
    ]
    if state.get("warnings"):
        context_parts.append("Warnings:\n" + "\n".join(state["warnings"]))

    messages = [
        SystemMessage(content=REPORT_SYSTEM_PROMPT),
        HumanMessage(content="\n\n".join(context_parts)),
    ]
    response = llm.invoke(messages)
    return {"report": response.content}


# ---------------------------------------------------------
# 5. CONDITIONAL EDGE
# ---------------------------------------------------------
def route_after_analyze(state: AgentState) -> str:
    return "human_review" if state.get("risk_level") == "high" else "generate_report"


# ---------------------------------------------------------
# 6. GRAPH ASSEMBLY
# ---------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("fetch_data", fetch_data)
    graph.add_node("analyze", analyze)
    graph.add_node("human_review", human_review)
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "fetch_data")
    graph.add_edge("fetch_data", "analyze")
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "human_review": "human_review",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("human_review", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run(question: str) -> dict:
    """Run the graph for a single question and return the final state."""
    initial_state: AgentState = {
        "input": question,
        "tools_called": [],
        "tool_results": {},
        "risk_level": "low",
        "warnings": [],
        "report": "",
    }
    return get_graph().invoke(initial_state)


if __name__ == "__main__":
    import sys

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("HATA: .env dosyasinda GROQ_API_KEY bulunamadi!")

    question = " ".join(sys.argv[1:]) or "What is the current price of Apple?"
    result = run(question)

    print(f"Question: {question}")
    print(f"Tools called: {result['tools_called']}")
    print(f"Risk level: {result['risk_level']}")
    print(f"Warnings: {result['warnings']}")
    print(f"\nReport:\n{result['report']}")
