import streamlit as st
from LLM_intent import classify_intent_llm
from symptom_extraction import extract_symptoms
from RAG import rag_answer

st.set_page_config(page_title="Medical NLP Assistant", layout="wide")

st.title("🧑🏻‍⚕️ Medical Question Assistant")

# Input box
user_question = st.text_area("Enter your medical question:", height=100)

if st.button("Run Analysis") and user_question.strip():

    with st.spinner("Analyzing..."):

        # 1. Intent Classification
        intent = classify_intent_llm(user_question)

        # 2. Symptom Extraction 
        extracted = extract_symptoms(user_question)

        # 3. RAG Final Answer
        answer, retrieved = rag_answer(user_question, k=5)

    # Display Results
    st.subheader("◻️ Intent Classification")
    st.write(f"**Predicted Intent:** `{intent}`")

    st.subheader("◻️ Symptom Extraction")
    if extracted:
        st.json(extracted)
    else:
        st.write("No symptoms extracted for this intent.")

    st.subheader("◻️ Retrieved Similar Questions (RAG)")
    for i, q in enumerate(retrieved, 1):
        st.write(f"**{i}.** {q}")

    st.subheader("🟩 Final Answer")
    st.write(answer)
