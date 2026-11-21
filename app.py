
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Python Tutor AI",
    page_icon="🐍",
    layout="centered"
)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./")
    model = AutoModelForCausalLM.from_pretrained("./")
    model.eval()
    return tokenizer, model

st.title("🐍 Python Tutor AI")
st.subheader("Your Personal AI Python Learning Assistant")

st.write(
    "Ask any beginner-level Python question "
    "and get an AI-generated explanation."
)

try:
    tokenizer, model = load_model()
    st.success("Python Tutor AI is ready!")

    question = st.text_input(
        "Enter your Python question:",
        placeholder="What is a variable in Python?"
    )

    if st.button("Generate Answer"):
        if question.strip():
            prompt = f"Question: {question} Answer:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            answer = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            st.subheader("AI Tutor Response")
            st.write(answer)
        else:
            st.warning("Please enter a question.")

except Exception as e:
    st.error(f"Model loading error: {e}")
