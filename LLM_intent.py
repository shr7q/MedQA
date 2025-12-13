from groq import Groq
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import classification_report
import json
from dotenv import load_dotenv
import os


# Load environment variables from .env file and Set up Groq API Key
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# Load dataset
df = pd.read_json("labels.json")
df.head()



# print all unique intent label categories
INTENT_LABELS = sorted(df["intent"].unique())
print("Here are the intent labels:", INTENT_LABELS)



# Intent Classification Function
def build_prompt(question):
    label_list = ", ".join(INTENT_LABELS)

    return f"""
You are an expert medical question classifier.

Classify the user's question into EXACTLY one of these intent categories:

{label_list}

Return ONLY valid JSON:
{{
    "intent": "one_of_the_categories"
}}

User question: "{question}"
"""


# Intent Classification Function

def classify_intent(question):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": build_prompt(question)}],
        temperature=0
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)["intent"]
    except:
        return "unknown"

if __name__ == "__main__":
    df = pd.read_json("labels.json")
    df_eval = df.sample(30, random_state=42).reset_index(drop=True)

    preds = []
    for q in tqdm(df_eval["question"]):
        preds.append(classify_intent(q))
        time.sleep(0.15)

    df_eval["pred_intent"] = preds

    print("\nClassification Report (LLM, 30 samples):")
    print(classification_report(df_eval["intent"], df_eval["pred_intent"]))

