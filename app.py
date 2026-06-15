import torch
import random
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from data import load_data, format_prompt


@st.cache_resource
def load_model():
    config = Config()
    torch.set_num_threads(config.torch_threads)
    model = AutoModelForCausalLM.from_pretrained(config.hub_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.hub_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, config


@st.cache_data
def load_test_data():
    config = Config()
    dataset = load_data(config)
    return dataset[config.test_split]


def generate_cypher(model, tokenizer, schema, question, config):
    example = {"schema": schema, "question": question, "cypher": ""}
    prompt = format_prompt(example)["text"]
    prompt = prompt.split("### Cypher:")[0] + "### Cypher:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# --- UI ---
st.title("Text2Cypher Demo")
st.caption("SmolLM2-135M fine-tuned to generate Cypher queries from natural language")

model, tokenizer, config = load_model()

tab1, tab2 = st.tabs(["Try it yourself", "Test set examples"])

# --- Tab 1: Custom input ---
with tab1:
    schema = st.text_area(
        "Graph Schema",
        value="Movie {title, year}, Person {name}, (Person)-[:DIRECTED]->(Movie)",
        height=80
    )
    question = st.text_input(
        "Question",
        value="Which movies did Christopher Nolan direct before 2010?"
    )

    if st.button("Generate Cypher"):
        with st.spinner("Generating..."):
            prediction = generate_cypher(model, tokenizer, schema, question, config)
        st.subheader("Generated Cypher")
        st.code(prediction, language="cypher")

# --- Tab 2: Random test examples ---
with tab2:
    if st.button("Load 3 Random Test Examples"):
        test_data = load_test_data()
        indices = random.sample(range(len(test_data)), 3)

        for i, idx in enumerate(indices):
            example = test_data[idx]
            with st.spinner(f"Generating example {i+1}..."):
                prediction = generate_cypher(
                    model, tokenizer,
                    example["schema"],
                    example["question"],
                    config
                )

            st.markdown(f"### Example {i+1}")
            st.markdown(f"**Schema:** `{example['schema']}`")
            st.markdown(f"**Question:** {example['question']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ground Truth**")
                st.code(example["cypher"], language="cypher")
            with col2:
                st.markdown("**Predicted**")
                st.code(prediction, language="cypher")

            st.divider()