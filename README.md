# Text-to-Cypher-LLM-Finetune
Finetuning a small LLM to generate Cypher queries from a natural language question and a graph schema.

## Setup & Run

### 1. Create Virtual Environment
```bash
python3 -m venv fvenv
```

### 2. Activate Virtual Environment
**Mac/Linux:**
```bash
source fvenv/bin/activate
```
**Windows:**
```bash
fvenv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

Login to HuggingFace:
```bash
hf auth login
```

## Reproduce

**Train:**
```bash
python train.py
```

**Evaluate:**

Per sample evaluation on Exact Match and Token F1 metric
Needs hugging face login

```bash
python evaluate.py
```
Results will be saved to `results/predictions.json`

**LLM Judge (SFT):**

ChatGPT 4o mini used to judge the structure 

```bash
python llm_judge.py --predictions results/predictions.json
```

## Metrics

Two primary metrics used deliberately:

- **Exact Match** — strict comparison, good baseline
- **Token F1** — measures token overlap, more forgiving than EM, more meaningful than BLEU for structured query language

**What they miss:**
- Neither executes the query against a real Neo4j DB
- EM penalizes semantically equivalent queries (`WHERE m.year < 2010` vs `WHERE 2010 > m.year`)
- Token F1 doesn't understand Cypher syntax — high score can still be a broken query
- Gold standard would be execution accuracy against a live DB — not feasible without data infrastructure

## LLM as a Judge

To get richer evaluation beyond Token F1, we use **Gemini Flash as a judge**. For each test sample it scores:

- **Structural Correctness (0-10)** — is the Cypher syntax and structure correct?
- **Semantic Equivalence (0-10)** — does it mean the same thing as ground truth?
- **Would Return Same Results** — binary yes/no, closest proxy to execution accuracy

This catches cases Token F1 misses — semantically equivalent queries, minor formatting differences, and structural errors that still score high on token overlap.

**Setup for llm judge:**

Create a `.env` file:

OPENAI_API_KEY=your-openai-api-key-here
Get an API key from platform.openai.com

```bash
pip install openai python-dotenv
```

**Run:**
```bash
# judge SFT predictions
python llm_judge.py --predictions results/predictions.json

```
Results saved to `results/llm_judge_results.json` with per-sample scores and explanations.

### Why this matters

Token F1 misses semantically equivalent queries — `WHERE m.year < 2010` vs `WHERE 2010 > m.year` scores 0 on Token F1 but should pass. LLM judge catches this. The **Would Return Same Results** metric is the closest approximation to execution accuracy without needing a live Neo4j database.

## Demo App

A simple Streamlit app to interact with the model.

```bash
pip install streamlit
streamlit run app.py
```

### Features
- **Try it yourself** — enter any schema and question, get a generated Cypher query. Optionally paste ground truth to see Exact Match and Token F1 scores.
- **Test set examples** — load 3 random examples from the test set with ground truth vs predicted Cypher side by side, with metrics.

### Example
Schema: `Movie {title, year}, Person {name}, (Person)-[:DIRECTED]->(Movie)`  
Question: `Which movies did Christopher Nolan direct before 2010?`  
Generated: `MATCH (p:Person {name: 'Christopher Nolan'})-[:DIRECTED]->(m:Movie) WHERE m.year < 2010 RETURN m.title`

## Design Decisions

**Full Fine-tune over LoRA**
LoRA is designed for large models (7B+) where full fine-tune is infeasible. SmolLM2 at 135M params is small enough to fully fine-tune on a single GPU in 21 minutes. Full fine-tune lets every weight adapt to the Cypher generation task. LoRA adds complexity around rank selection and adapter merging without any memory benefit at this scale.

**SmolLM2-Instruct**
The Instruct variant already understands structured prompt formats from pretraining. Our Schema, Question, Cypher format aligns with this so we leverage existing instruction following capability rather than teaching prompt structure from scratch.

**Max length 1024**
Complex graph schemas alone consume 500+ tokens. Lower values truncate the schema mid-way and lose node and relationship information that is critical for generating correct queries.

**Deterministic generation**
Cypher generation is a deterministic task so sampling introduces randomness that hurts reliability. Greedy decoding ensures the same question always gives the same query, making evaluation reproducible and consistent.

**Dynamic padding**
Padding all sequences to max length upfront wastes memory. Dynamic padding per batch reduces memory usage significantly and speeds up training without affecting model quality.

**Metrics: Exact Match and Token F1 over BLEU**
BLEU measures n-gram order and is designed for natural language where word order matters. Cypher order often does not since `WHERE a=1 AND b=2` is identical to `WHERE b=2 AND a=1`. Token F1 with sets ignores order and measures token overlap which is more appropriate for structured query language.

**LLM as a Judge**
Token F1 and Exact Match compare strings not semantics. Two queries can be written differently but return identical results from the database. GPT-4o-mini evaluates structural correctness, semantic equivalence, and whether queries would return the same results which is the closest approximation to execution accuracy without a live Neo4j database.

## Limitations

**Model capacity**
135M parameters is small for a code generation task. The model learns common Cypher patterns well but struggles with complex multi-hop queries and schema structures it has not seen during training.

**No execution validation**
The gold standard for text-to-query tasks is execution accuracy where you run both queries against a real database and compare results. Without a live Neo4j database we approximate this with the LLM judge but it is not the same thing.

**Generation stopping**
The model does not reliably generate an EOS token so it requires an explicit stopping strategy at inference time. The proper fix requires retraining with an explicit end marker in the prompt format.

**Dataset bias**
Trained on a single curated dataset, the model is biased toward the schema patterns and query styles present in that data. It may generalize poorly to different graph domains or naming conventions.