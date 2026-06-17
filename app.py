import torch
import random
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from data import load_data, format_prompt, generate_cypher
from metrics import compute_metrics


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


# --- session state init ---
if "tab1_prediction" not in st.session_state:
    st.session_state.tab1_prediction = None

if "tab2_examples" not in st.session_state:
    st.session_state.tab2_examples = None


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
    ground_truth = st.text_input(
        "Ground Truth (optional — paste to compare metrics)"
    )

    if st.button("Generate Cypher"):
        with st.spinner("Generating..."):
            st.session_state.tab1_prediction = generate_cypher(
                model, tokenizer, schema, question, config
            )

    if st.session_state.tab1_prediction:
        st.subheader("Generated Cypher")
        st.text_area(
            "Generated Cypher Output",
            value=st.session_state.tab1_prediction,
            height=120,
            disabled=True,
            label_visibility="collapsed"
        )

        if ground_truth:
            st.subheader("Metrics")
            metrics = compute_metrics(st.session_state.tab1_prediction, ground_truth)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Exact Match", "✅" if metrics["exact_match"] else "❌")
            with col2:
                st.metric("Token F1", metrics["token_f1"])

# --- Tab 2: Random test examples ---
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        load_clicked = st.button("Load 3 Random Test Examples")
    with col2:
        refresh_clicked = st.button("🔄 Refresh Examples")

    if refresh_clicked:
        st.session_state.tab2_examples = None
        st.rerun()

    if load_clicked and st.session_state.tab2_examples is None:
        test_data = load_test_data()
        indices = random.sample(range(len(test_data)), 3)
        results = []

        for idx in indices:
            example = test_data[idx]
            with st.spinner("Generating..."):
                prediction = generate_cypher(
                    model, tokenizer,
                    example["schema"],
                    example["question"],
                    config
                )
            results.append({
                "schema": example["schema"],
                "question": example["question"],
                "ground_truth": example["cypher"],
                "prediction": prediction,
                "metrics": compute_metrics(prediction, example["cypher"])
            })

        st.session_state.tab2_examples = results

    if st.session_state.tab2_examples:
        for i, ex in enumerate(st.session_state.tab2_examples):
            st.markdown(f"### Example {i+1}")
            st.markdown(f"**Schema:** `{ex['schema']}`")
            st.markdown(f"**Question:** {ex['question']}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ground Truth**")
                st.text_area(
                    f"Ground Truth {i}",
                    value=ex["ground_truth"],
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"gt_{i}"
                )
            with col2:
                st.markdown("**Predicted**")
                st.text_area(
                    f"Predicted {i}",
                    value=ex["prediction"],
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"pred_{i}"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Exact Match", "✅" if ex["metrics"]["exact_match"] else "❌")
            with col2:
                st.metric("Token F1", ex["metrics"]["token_f1"])

            st.divider()