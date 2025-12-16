from groq import Groq
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import classification_report
import json
from dotenv import load_dotenv
import os

# Load environment variables 
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# Load dataset
df = pd.read_json("labels.json")


# MERGED INTENT MAPPING 
INTENT_MAP = {
    #  Symptom related
    "symptom_centric_query": "symptom_question",
    "severity_question": "symptom_question",

    # Disease info
    "disease_general_health_information": "disease_information",

    # Prognosis
    "prognosis_inquiry": "prognosis_question",

    # Keep as is
    "treatment_question": "treatment_question",
    "causality_question": "causality_question",
    "transmission_question": "transmission_question",
    "diagnosis_decision_question": "diagnosis_decision_question",
}

# Apply mapping to ground truth
df["intent_merged"] = df["intent"].map(INTENT_MAP)

# Sanity check
print("Merged intent labels:")
print(sorted(df["intent_merged"].unique()))

INTENT_LABELS = sorted(df["intent_merged"].unique())


# LLM prompt
def build_prompt(question):
    label_list = ", ".join(INTENT_LABELS)

    return f"""
You are an expert medical question classifier.

Classify the user's question into EXACTLY ONE of the following intent categories:

{label_list}

Return ONLY valid JSON in this format:
{{
  "intent": "one_of_the_categories"
}}

User question: "{question}"
"""

def classify_intent_llm(question):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": build_prompt(question)}],
            temperature=0
        )

        content = response.choices[0].message.content

        # Parse JSON robustly
        try:
            return json.loads(content)["intent"]
        except:
            start = content.find("{")
            end = content.rfind("}") + 1
            cleaned = content[start:end]
            return json.loads(cleaned).get("intent", "unknown")

    except Exception as e:
        print("LLM error:", e)
        return "unknown"

# Evaluation
if __name__ == "__main__":

    # Sample evaluation set
    df_eval = df.sample(30, random_state=42).reset_index(drop=True)

    preds = []
    for q in tqdm(df_eval["question"], desc="Classifying"):
        preds.append(classify_intent_llm(q))
        time.sleep(0.15)  # avoid rate limits

    df_eval["pred_intent_raw"] = preds

    # Map predicted intents to merged labels
    df_eval["pred_intent_merged"] = df_eval["pred_intent_raw"].map(
        lambda x: INTENT_MAP.get(x, x)
    )

    print("\nClassification Report (LLM, merged labels):\n")

    print(
        classification_report(
            df_eval["intent_merged"],
            df_eval["pred_intent_merged"],
            labels=INTENT_LABELS,
            zero_division=0
        )
    )

