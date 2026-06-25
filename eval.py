"""
Evaluation Script for the LangGraph Finance Agent
---------------------------------------------------
Runs a fixed set of test questions through langgraph_agent.run(), logs which
tool(s) were called, and scores each response with an LLM-as-Judge (a second
Groq/Llama model) on accuracy, relevance, and clarity (1-5 each).

Output:
- eval_results.csv : per-question results
- Console summary  : average overall score + tool-selection accuracy %
"""

from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "finance-agent-eval"
# LANGCHAIN_API_KEY zaten .env'den load_dotenv() ile geliyor, tekrar set etme.

import csv
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph_agent import run as run_agent

# A smaller/different Llama model acts as the judge so it's not grading
# its own homework with the exact same weights as the agent.
JUDGE_MODEL = "llama-3.1-8b-instant"

TEST_CASES = [
    {"question": "What is the current price of Apple?", "expected_tools": ["get_stock_price"]},
    {"question": "Compare the PE ratio of Google and Meta.", "expected_tools": ["get_company_fundamentals"]},
    {"question": "Give me a fundamental analysis of Tesla.", "expected_tools": ["get_company_fundamentals"]},
    {"question": "What's today's daily change percentage for Microsoft stock?", "expected_tools": ["get_stock_price"]},
    {"question": "Tell me about Amazon's market cap and sector.", "expected_tools": ["get_company_fundamentals"]},
    {"question": "How is THYAO.IS performing today?", "expected_tools": ["get_stock_price"]},
    {"question": "What is the 52 week high for Nvidia?", "expected_tools": ["get_company_fundamentals"]},
    {"question": "Compare the current prices of Apple and Microsoft.", "expected_tools": ["get_stock_price"]},
    {"question": "Based on today's price movement, is Tesla stock volatile right now?", "expected_tools": ["get_stock_price"]},
    {"question": "What sector does Netflix operate in?", "expected_tools": ["get_company_fundamentals"]},
    {"question": "Give me both the price and fundamentals for Coca-Cola.", "expected_tools": ["get_stock_price", "get_company_fundamentals"]},
    {"question": "What is the market cap of Meta?", "expected_tools": ["get_company_fundamentals"]},
    {"question": "Give me the current price of Google stock.", "expected_tools": ["get_stock_price"]},
    {"question": "Provide a full analysis (price and fundamentals) of Amazon.", "expected_tools": ["get_stock_price", "get_company_fundamentals"]},
]

JUDGE_PROMPT = """You are an impartial evaluator of an AI financial analyst's response.

Question asked by the user:
{question}

Agent's response:
{response}

Rate the response on a scale of 1-5 (5 = best) for EACH of the following criteria:
- accuracy: Does the response use concrete data and avoid making up numbers?
- relevance: Does the response directly address the user's question?
- clarity: Is the response clear, well-organized, and easy to understand?

Respond in EXACTLY this format (no extra text, no markdown):
accuracy: <score>
relevance: <score>
clarity: <score>
overall: <score>
reasoning: <one short sentence>
"""


def judge_response(question: str, response: str, judge_llm) -> dict:
    messages = [
        SystemMessage(content="You are a strict, impartial evaluator. Follow the requested output format exactly."),
        HumanMessage(content=JUDGE_PROMPT.format(question=question, response=response)),
    ]
    text = judge_llm.invoke(messages).content

    scores = {"accuracy": None, "relevance": None, "clarity": None, "overall": None, "reasoning": ""}
    for line in text.splitlines():
        line = line.strip()
        for key in ("accuracy", "relevance", "clarity", "overall"):
            if line.lower().startswith(f"{key}:"):
                match = re.search(r"(\d+)", line)
                if match:
                    scores[key] = int(match.group(1))
        if line.lower().startswith("reasoning:"):
            scores["reasoning"] = line.split(":", 1)[1].strip()

    return scores


def tool_selection_correct(expected: list, actual: list) -> bool:
    """True if every expected tool was called at least once (extra calls are OK)."""
    return set(expected).issubset(set(actual))


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("HATA: .env dosyasinda GROQ_API_KEY bulunamadi!")

    judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

    rows = []
    overall_scores = []
    tool_matches = 0

    for i, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        expected_tools = case["expected_tools"]

        print(f"[{i}/{len(TEST_CASES)}] {question}")

        try:
            result = run_agent(question)
            tools_called = result.get("tools_called", [])
            report = result.get("report", "")
            risk_level = result.get("risk_level", "unknown")
        except Exception as e:
            print(f"  ERROR running agent: {e}")
            tools_called, report, risk_level = [], f"ERROR: {e}", "unknown"

        match = tool_selection_correct(expected_tools, tools_called)
        if match:
            tool_matches += 1
        print(f"  tools_called={tools_called}  expected={expected_tools}  match={match}  risk={risk_level}")

        try:
            judge = judge_response(question, report, judge_llm)
        except Exception as e:
            print(f"  ERROR judging response: {e}")
            judge = {"accuracy": None, "relevance": None, "clarity": None, "overall": None, "reasoning": f"judge error: {e}"}

        if judge.get("overall") is not None:
            overall_scores.append(judge["overall"])
        print(f"  judge={judge}")

        rows.append({
            "question": question,
            "expected_tools": ";".join(expected_tools),
            "tools_called": ";".join(tools_called),
            "tool_selection_correct": match,
            "risk_level": risk_level,
            "accuracy": judge.get("accuracy"),
            "relevance": judge.get("relevance"),
            "clarity": judge.get("clarity"),
            "overall": judge.get("overall"),
            "judge_reasoning": judge.get("reasoning"),
            "response": report,
        })

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    tool_accuracy = (tool_matches / len(TEST_CASES)) * 100

    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    print(f"Total questions: {len(TEST_CASES)}")
    print(f"Average judge score (overall, 1-5): {avg_overall:.2f}")
    print(f"Tool selection accuracy: {tool_accuracy:.1f}% ({tool_matches}/{len(TEST_CASES)})")
    print(f"Results written to: {csv_path}")


if __name__ == "__main__":
    main()
