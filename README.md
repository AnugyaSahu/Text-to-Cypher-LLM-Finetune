# Text-to-Cypher-LLM-Finetune
Finetuning a small LLM to generate Cypher queries from a natural language question and a graph schema.

## Setup & Run

### 1. Create Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment
**Mac/Linux:**
```bash
source venv/bin/activate
```
**Windows:**
```bash
venv\Scripts\activate
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

## Design Decisions and Limitations

**Full fine-tune over LoRA:**
SmolLM2 is 135M params — small enough for full fine-tune on CPU without memory issues. LoRA adds complexity without clear benefit at this scale.

**Prompt format:**
Used instruction-style format matching SmolLM2-Instruct's pretraining format:
Schema:
Question:
Cypher:

Matching the expected format leverages pretraining and gives better results.

**Max length:**
Set to 1024 tokens after finding that complex schemas alone consume 500+ tokens. Increasing from 256 to 1024 improved Token F1 from 0.41 to 0.64.

**Deterministic generation:**
Used `do_sample=False` for reproducibility — same input always gives same output, making evaluation reliable.

**Post-processing:**
Model sometimes repeats the prompt template after generating the query. Added a post-processing step to strip everything after `### Cypher:` appears in the output.

**Limitations:**
- 135M model learns Cypher syntax patterns but not semantic reasoning
- No query execution validation against a real Neo4j database
- Model struggles with complex multi-hop queries and unseen schema structures
