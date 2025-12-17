# 🧠 MedQA: Modular NLP System for Medical Question Understanding

## Overview
MedQA is a modular Natural Language Processing (NLP) system designed to analyze and respond to medical questions in a **safe, interpretable, and structured** manner.

Instead of directly generating answers, the system decomposes a user query into intermediate steps—**intent classification**, **symptom extraction**, and **retrieval-augmented generation (RAG)**—before producing a final response.

This project was developed as a final project for an NLP course, with emphasis on **system design, evaluation, and interpretability**, rather than purely model performance.

---

## Key Features
- Intent classification using Large Language Models (LLMs)
- Structured symptom extraction (symptom, body location, duration, trigger)
- Retrieval-Augmented Generation (RAG) using FAISS + sentence embeddings
- Modular architecture with independently evaluable components
- Ethically conscious design focused on safety and transparency
- Optional Streamlit frontend for interactive demonstration

---

## System Architecture
The pipeline follows four main stages:

1. **Intent Classification**  
   Classifies a medical question into one of several merged intent categories (e.g., symptom question, treatment question, disease information).

2. **Symptom Extraction (Conditional)**  
   For symptom-related intents, extracts structured attributes such as symptom name, body location, duration, and trigger.

3. **Retrieval-Augmented Generation (RAG)**  
   Retrieves semantically similar medical questions from a large unlabeled corpus and uses them as contextual grounding.

4. **Final Answer Generation**  
   Produces a medically safe, non-diagnostic response using an LLM, while exposing intermediate outputs for transparency.

---

## Dataset
- **HealthSearchQA** dataset (via HuggingFace)
- ~3,000+ unlabeled medical questions used as the RAG retrieval corpus
- 150 manually labeled questions used for intent classification and symptom extraction evaluation
- Original 18 intent labels merged into 10 higher-level categories to reduce sparsity

---

## Models & Tools
- **LLMs:** Groq-hosted LLaMA models (e.g., LLaMA 3.x)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Search:** FAISS (CPU)
- **Evaluation:** scikit-learn (classification metrics)
- **Frontend (optional):** Streamlit

---

## Repository Structure
```
MedQA/
├── app.py # Streamlit frontend
├── intent_classification.py # Intent classification module
├── symptom_extraction.py # Symptom extraction module
├── rag_pipeline.py # RAG retrieval + answer generation
├── data/
│ └── corpus.json # Complete HealthQA dataset
| └── labels.json # Labeled subset
├── baselines.ipynb
├── requirements.txt
├── README.md
└── Setup.md
```
