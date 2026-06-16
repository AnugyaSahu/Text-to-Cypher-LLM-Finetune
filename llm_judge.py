import json
import time
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from metrics import compute_metrics


JUDGE_PROMPT = """You are an expert in Neo4j Cypher query language.

You will be given:
- A graph schema
- A natural language question
- A ground truth Cypher query
- A predicted Cypher query

Evaluate the predicted query and return ONLY a JSON object with no extra text, no markdown, no backticks.

Schema: {schema}

Question: {question}

Ground Truth: {ground_truth}

Predicted: {prediction}

Return this exact JSON format:
{{
  "structural_correctness": <0-10>,
  "semantic_equivalence": <0-10>,
  "would_return_same_results": <true or false>,
  "explanation": "<one sentence>"
}}"""


def judge_single(client, schema, question, ground_truth, prediction):
    prompt = JUDGE_PROMPT.format(
        schema=schema,
        question=question,
        ground_truth=ground_truth,
        prediction=prediction
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

        # clean any markdown backticks just in case
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        result = json.loads(text)
        return result

    except Exception as e:
        print(f"Judge error: {e}")
        return {
            "structural_correctness": 0,
            "semantic_equivalence": 0,
            "would_return_same_results": False,
            "explanation": f"Error: {str(e)}"
        }


def run_llm_judge(predictions_path: str, api_key: str, output_path: str = "results/llm_judge_results.json"):

    client = OpenAI(api_key=api_key)

    # load predictions
    with open(predictions_path, "r") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Judging {len(samples)} samples...")

    results = []
    total_structural = 0
    total_semantic = 0
    total_same_results = 0

    for i, sample in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}")

        judgment = judge_single(
            client,
            sample["schema"],
            sample["question"],
            sample["ground_truth"],
            sample["prediction"]
        )

        # also compute our own metrics for comparison
        token_metrics = compute_metrics(sample["prediction"], sample["ground_truth"])

        total_structural += judgment["structural_correctness"]
        total_semantic += judgment["semantic_equivalence"]
        total_same_results += int(judgment["would_return_same_results"])

        results.append({
            "question": sample["question"],
            "schema": sample["schema"],
            "ground_truth": sample["ground_truth"],
            "prediction": sample["prediction"],
            "token_f1": token_metrics["token_f1"],
            "exact_match": token_metrics["exact_match"],
            "llm_structural": judgment["structural_correctness"],
            "llm_semantic": judgment["semantic_equivalence"],
            "llm_same_results": judgment["would_return_same_results"],
            "llm_explanation": judgment["explanation"]
        })

        # sleep to avoid rate limiting
        time.sleep(1)

    n = len(samples)
    summary = {
        "aggregate": {
            "token_f1": round(data["aggregate"]["token_f1"], 4),
            "exact_match": round(data["aggregate"]["exact_match"], 4),
            "llm_structural_avg": round(total_structural / n, 2),
            "llm_semantic_avg": round(total_semantic / n, 2),
            "llm_same_results_pct": round(total_same_results / n * 100, 2),
            "total_samples": n
        },
        "samples": results
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- LLM Judge Summary ---")
    print(f"Token F1:                  {summary['aggregate']['token_f1']}")
    print(f"Exact Match:               {summary['aggregate']['exact_match']}")
    print(f"LLM Structural Score:      {summary['aggregate']['llm_structural_avg']} / 10")
    print(f"LLM Semantic Score:        {summary['aggregate']['llm_semantic_avg']} / 10")
    print(f"Would Return Same Results: {summary['aggregate']['llm_same_results_pct']}%")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="results/predictions.json")
    parser.add_argument("--output", type=str, default="results/llm_judge_results.json")
    args = parser.parse_args()

    run_llm_judge(
        predictions_path=args.predictions,
        api_key=API_KEY,
        output_path=args.output
    )