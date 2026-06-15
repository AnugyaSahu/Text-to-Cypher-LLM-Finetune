# Text-to-Cypher-LLM-Finetune
Finetuning a small LLM to generate Cypher queries from a natural language question and a graph schema

## Setup

```bash
git clone https://github.com/your-username/text2cypher-finetune
cd text2cypher-finetune
pip install -r requirements.txt
```

Login to HuggingFace:
```bash
huggingface-cli login
```

## Reproduce

**Train:**
```bash
python train.py
```

**Evaluate:**
```bash
python evaluate.py
```

Results will be saved to `results/predictions.json`

## Metrics

Two metrics used deliberately:

- **Exact Match** — strict comparison, good baseline
- **Token F1** — measures token overlap, more forgiving than EM, more meaningful than BLEU for structured query language

**What they miss:**
- Neither executes the query against a real Neo4j DB
- EM penalizes semantically equivalent queries (`WHERE m.year < 2010` vs `WHERE 2010 > m.year`)
- Token F1 doesn't understand Cypher syntax — high score can still be a broken query
- Gold standard would be execution accuracy against a live DB — not feasible without data infrastructure

## Design Decisions and Limitations

**Full fine-tune over LoRA:**
SmolLM2 is 135M params — small enough for full fine-tune on CPU without memory issues. LoRA adds complexity without clear benefit at this scale.

**Prompt format:**
Used instruction-style format matching SmolLM2-Instruct's pretraining format:
Schema:
Question:
Cypher:

Matching the expected format leverages pretraining and gives better results.

**Deterministic generation:**
Used `do_sample=False` for reproducibility — same input always gives same output, making evaluation reliable.

**Limitations:**
- No query execution validation
- CPU training limits batch size and epochs
- Model may struggle with complex schemas or multi-hop queries