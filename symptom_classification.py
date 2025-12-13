from groq import Groq
from dotenv import load_dotenv
import pandas as pd
import json
import os


# Load environment variables from .env file and Set up Groq API Key
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)


# Define body part groups for symptom classification
BODY_GROUPS = {
    "head": [
        "brain", "head", "face", "eye", "eyes", "lips", "mouth",
        "nose", "gums", "tongue", "teeth", "salivary_glands",
        "psychological"
    ],
    "neck_throat": [
        "throat"
    ],
    "chest": [
        "chest", "lungs", "heart"
    ],
    "abdomen": [
        "abdomen", "bladder", "kidney", "rectum", "prostate", "uterus",
        "urinary system", "urinary_tract", "reproductive system",
        "testicle", "genital", "penis"
    ],
    "back_spine": [
        "back", "spine", "hips", "joints"
    ],
    "upper_limb": [
        "hand"
    ],
    "skin_hair": [
        "skin", "hair", "body hair", "tissue"
    ],
    "circulatory_blood": [
        "blood", "systemic", "nervous system", "nervous_system"
    ],
    "general_body": [
        "body"
    ]
}


# Map specific body location to broader anatomical group
def map_body_location(loc):
    if not loc:
        return ""

    loc = loc.lower().strip()
    for group, members in BODY_GROUPS.items():
        if loc in members:
            return group

    return "other"



# Prompt builder 
def symptom_prompt(question):
    return f"""
You are a medical symptom extraction system. 
Extract the following fields from the user's question:

- "symptom": short phrase describing what the user is experiencing
- "body_location": the explicit body part mentioned (use lowercase)
- "duration": when or how long the symptom occurs (if present)
- "trigger": what causes, worsens, or relates to the symptom (if present)

Return ONLY valid JSON. Do NOT add commentary.

Examples:

Q: "Why do my eyes feel dry in the morning?"
A: {{"symptom": "dry eyes", "body_location": "eyes", "duration": "in the morning", "trigger": ""}}

Q: "I get knee pain when running."
A: {{"symptom": "knee pain", "body_location": "knee", "duration": "", "trigger": "running"}}

Q: "My chest feels tight when I wake up."
A: {{"symptom": "chest tightness", "body_location": "chest", "duration": "when I wake up", "trigger": ""}}

Now extract for this question:

Q: "{question}"
A:
"""

# Symptom Extraction Function
def extract_symptoms(question, model="llama-3.1-8b-instant"):
    prompt = symptom_prompt(question)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content  
        
        if isinstance(raw, list):
            raw = "".join([part.text for part in raw])

        # Parse JSON safely
        data = json.loads(raw)

        # Map anatomical group
        data["body_location_group"] = map_body_location(data.get("body_location", ""))

        return data

    except Exception as e:
        print("Error:", e)
        return {
            "symptom": "",
            "body_location": "",
            "duration": "",
            "trigger": "",
            "body_location_group": "other"
        }

if __name__ == "__main__":
    # Evaluation on Gold standard data that has been validated manually
    gold_data = [
    {
        "question": "Are dry lips a symptom of anything?",
        "symptom": "dry lips",
        "body_location": "lips",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Are floaters in eye serious?",
        "symptom": "floaters in eye",
        "body_location": "eye",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Are red veins in eyes serious?",
        "symptom": "red veins in eyes",
        "body_location": "eyes",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "At what age is occasional shortness of breath normal?",
        "symptom": "shortness of breath",
        "body_location": "chest",
        "duration": "occasional",
        "trigger": ""
    },
    {
        "question": "At what age is rectal bleeding common?",
        "symptom": "rectal bleeding",
        "body_location": "rectum",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Can high blood pressure cause blue lips?",
        "symptom": "blue lips",
        "body_location": "lips",
        "duration": "",
        "trigger": "high blood pressure"
    },
    {
        "question": "Can dizziness be serious?",
        "symptom": "dizziness",
        "body_location": "brain",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Can dry eye syndrome be fixed?",
        "symptom": "dry eye syndrome",
        "body_location": "eye",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Can rectal bleeding be serious?",
        "symptom": "rectal bleeding",
        "body_location": "rectum",
        "duration": "",
        "trigger": ""
    },
    {
        "question": "Can runny nose be a symptom of Covid?",
        "symptom": "runny nose",
        "body_location": "nose",
        "duration": "",
        "trigger": "Covid infection"
    }
    ]

    # Create DataFrame from gold data
    gold_df = pd.DataFrame(gold_data)

    # Generate predictions
    predictions = []

    # Loop through each question and extract symptoms
    for q in gold_df["question"]:
        pred = extract_symptoms(q)  # your working function
        predictions.append(pred)

    # Create DataFrame from predictions
    pred_df = pd.DataFrame(predictions)

    # Combine gold and predicted for evaluation
    eval_df = pd.concat([gold_df, pred_df.add_prefix("pred_")], axis=1)


    # Calculate Exact Match Accuracy
    def exact_match(a, b):
        a = "" if pd.isna(a) else str(a)
        b = "" if pd.isna(b) else str(b)
        return a.strip().lower() == b.strip().lower()

    accuracy_results = {
        "symptom_accuracy": eval_df.apply(lambda r: exact_match(r["symptom"], r["pred_symptom"]), axis=1).mean(),
        "location_accuracy": eval_df.apply(lambda r: exact_match(r["body_location"], r["pred_body_location"]), axis=1).mean(),
        "duration_accuracy": eval_df.apply(lambda r: exact_match(r["duration"], r["pred_duration"]), axis=1).mean(),
        "trigger_accuracy": eval_df.apply(lambda r: exact_match(r["trigger"], r["pred_trigger"]), axis=1).mean(),
    }
    print(accuracy_results)


    def f1_token(gold, pred):
        gold = "" if pd.isna(gold) else str(gold)
        pred = "" if pd.isna(pred) else str(pred)

        gold_tokens = set(gold.lower().split())
        pred_tokens = set(pred.lower().split())

        tp = len(gold_tokens & pred_tokens)
        fp = len(pred_tokens - gold_tokens)
        fn = len(gold_tokens - pred_tokens)

        if tp == 0:
            return 0.0

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2 * precision * recall / (precision + recall + 1e-9)

    eval_df["f1_symptom"] = eval_df.apply(lambda r: f1_token(r["symptom"], r["pred_symptom"]), axis=1)
    eval_df["f1_location"] = eval_df.apply(lambda r: f1_token(r["body_location"], r["pred_body_location"]), axis=1)
    eval_df["f1_trigger"]  = eval_df.apply(lambda r: f1_token(r["trigger"], r["pred_trigger"]), axis=1)
    eval_df["f1_duration"] = eval_df.apply(lambda r: f1_token(r["duration"], r["pred_duration"]), axis=1)

    print(eval_df[["f1_symptom", "f1_location", "f1_trigger", "f1_duration"]].mean())

