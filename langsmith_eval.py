"""
LangSmith Dataset + Evaluation for the Finance Agent
-----------------------------------------------------
Uploads the 14 questions/responses from eval_results.csv as a LangSmith
Dataset, then runs a LangSmith experiment against them with an
LLM-as-Judge evaluator (accuracy, relevance, clarity -> 1-5 quality score).
"""

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"

import csv
import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import Client, evaluate

DATASET_NAME = "finance-agent-questions"
EXPERIMENT_NAME = "finance-agent-v2"
CSV_PATH = "eval_results.csv"
JUDGE_MODEL = "llama-3.1-8b-instant"

client = Client()
judge_llm = ChatGroq(model=JUDGE_MODEL, temperature=0)

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


def load_examples(csv_path: str) -> list[dict]:
    examples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(
                {
                    "inputs": {"question": row["question"]},
                    "outputs": {"response": row["response"]},
                }
            )
    return examples


def get_or_create_dataset(name: str):
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(dataset_name=name)


def upload_dataset(examples: list[dict]):
    dataset = get_or_create_dataset(DATASET_NAME)
    client.create_examples(
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        dataset_id=dataset.id,
    )
    return dataset


def llm_judge_quality(inputs: dict, outputs: dict) -> dict:
    """LLM-as-Judge evaluator: rates the response 1-5 on accuracy, relevance, clarity."""
    question = (inputs or {}).get("question", "")
    response = (outputs or {}).get("response", "")

    messages = [
        SystemMessage(content="You are a strict, impartial evaluator. Follow the requested output format exactly."),
        HumanMessage(content=JUDGE_PROMPT.format(question=question, response=response)),
    ]
    text = judge_llm.invoke(messages).content

    overall, reasoning = None, ""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("overall:"):
            match = re.search(r"(\d+)", line)
            if match:
                overall = int(match.group(1))
        if line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    return {"key": "llm_judge_quality", "score": overall, "comment": reasoning}


if __name__ == "__main__":
    examples = load_examples(CSV_PATH)
    response_by_question = {ex["inputs"]["question"]: ex["outputs"]["response"] for ex in examples}

    dataset = get_or_create_dataset(DATASET_NAME)
    existing = list(client.list_examples(dataset_id=dataset.id))
    if not existing:
        upload_dataset(examples)
        print(f"Uploaded {len(examples)} examples to dataset '{DATASET_NAME}'.")
    else:
        print(f"Dataset '{DATASET_NAME}' already has {len(existing)} examples, skipping upload.")

    def target(inputs: dict) -> dict:
        """The 'system under test' for the experiment: replays the stored response."""
        return {"response": response_by_question.get(inputs["question"], "")}

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[llm_judge_quality],
        experiment_prefix=EXPERIMENT_NAME,
    )

    print(f"Experiment '{EXPERIMENT_NAME}' complete. View results in the LangSmith UI.")
