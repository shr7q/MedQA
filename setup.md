# MedQA Setup Guide

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Environment Configuration](#environment-configuration)
5. [Running the Application](#running-the-application)
6. [Running Individual Components](#running-individual-components)
7. [Project Structure](#project-structure)
8. [Data Files](#data-files)
9. [Troubleshooting](#troubleshooting)
10. [Additional Notes](#additional-notes)

---

## Project Overview

**MedQA** is a modular natural language processing pipeline designed for medical question understanding. The system integrates three primary components:

- **Intent Classification** - LLM-based zero-shot learning using Llama-3.3-70B
- **Symptom Extraction** - Few-shot prompting methodology with Llama-3.1-8B
- **Retrieval-Augmented Generation (RAG)** - FAISS semantic search combined with LLM generation

The system processes free-form health queries and generates safe, contextual responses while maintaining transparency through modular component evaluation.

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: Version 3.8 or higher (tested with Python 3.10+)
- **RAM**: Minimum 4GB (8GB recommended for optimal performance)
- **Storage**: At least 500MB available space for dependencies and models
- **Internet Connection**: Required for initial setup and API calls

### Required Accounts

- **Groq API Account**: Free account available at [console.groq.com](https://console.groq.com)
  - Provides access to Llama language models
  - Free tier includes rate limits suitable for research purposes
  - No credit card required for registration

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/shr7q/MedQA.git
cd MedQA
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

**Using conda (alternative):**
```bash
# Create conda environment
conda create -n medqa python=3.10

# Activate environment
conda activate medqa
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Dependencies installed:**
- `streamlit` - Web interface framework
- `groq` - Groq API client for LLM access
- `sentence-transformers` - Text embedding generation
- `faiss-cpu` - Facebook AI Similarity Search (vector database)
- `numpy` - Numerical computing library
- `pandas` - Data manipulation and analysis
- `transformers` - Hugging Face transformers library
- `tqdm` - Progress bar visualization for iterations
- `python-dotenv` - Environment variable management (installed with groq)
- `scikit-learn` - Machine learning utilities (for baseline evaluation)

**Estimated Installation Time:** 2-5 minutes depending on network speed

---

## Environment Configuration

### Step 1: Obtain Groq API Key

1. Navigate to [console.groq.com](https://console.groq.com)
2. Register for a free account (no credit card required)
3. Access the **API Keys** section
4. Select **"Create API Key"**
5. Copy the generated API key (format: `gsk_...`)

### Step 2: Create Environment File

Create a `.env` file in the project root directory:

```bash
# Create .env file (Linux/macOS)
touch .env

# Create .env file (Windows)
type nul > .env
```

### Step 3: Configure API Key

Open the `.env` file in a text editor and add:

```env
GROQ_API_KEY=your_actual_api_key_here
```

**Example:**
```env
GROQ_API_KEY=gsk_abc123xyz456def789ghi012jkl345mno678pqr901stu234vwx567yz
```

**Security Note:**
- Never commit `.env` files to version control systems (already included in `.gitignore`)
- Maintain confidentiality of API keys
- Do not share `.env` files with unauthorized parties
- Regenerate API key immediately if compromised

### Step 4: Verify Environment Setup

```bash
# Verify .env file exists
ls -la .env

# Verify environment variables load correctly
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', 'Yes' if os.getenv('GROQ_API_KEY') else 'No')"
```

Expected output: `API Key loaded: Yes`

---

## Running the Application

### Interactive Web Interface (Recommended)

The main application provides a user interface via Streamlit for the complete pipeline.

```bash
# Ensure current directory is MedQA
cd MedQA

# Launch Streamlit application
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.xxx:8501
```

**Usage Instructions:**
1. Open web browser and navigate to `http://localhost:8501`
2. Enter a medical question in the provided text area
3. Select **"Run Analysis"** button
4. Review results:
   - **Intent Classification**: Predicted intent category
   - **Symptom Extraction**: Structured symptom attributes (when applicable)
   - **Retrieved Questions**: Top 5 similar questions from corpus
   - **Final Answer**: RAG-generated response

**Example queries for testing:**
- "Are boils and carbuncles curable?"
- "Why do my eyes feel dry in the morning?"
- "Can stress cause headaches?"
- "What are the symptoms of diabetes?"

**To terminate the server:**
- Press `Ctrl+C` in the terminal window

---

## Running Individual Components

Each component can be executed independently for testing and evaluation purposes.

### 1. Intent Classification

**Purpose:** Classify medical questions into predefined intent categories

```bash
python LLM_intent.py
```

**Functionality:**
- Loads 148 labeled questions from `labels.json`
- Randomly samples 30 questions (random seed=42 for reproducibility)
- Classifies each question using Llama-3.3-70B (zero-shot approach)
- Outputs classification report including precision, recall, and F1-score

**Estimated Runtime:** Approximately 45 seconds (30 questions at ~1.5s per query)

**Expected output:**
```
Here are the intent labels: ['ability_question', 'activity_question', ...]

100%|████████████████████| 30/30 [00:45<00:00,  1.52s/it]

Classification Report (LLM, 30 samples):
              precision    recall  f1-score   support

   treatment_question       1.00      0.95      0.97        20
   severity_question        0.92      1.00      0.96        11
   ...
   
   accuracy                           0.77        30
   macro avg              0.64      0.66      0.64        30
   weighted avg           0.79      0.77      0.75        30
```

**Note:** Script includes 0.15-second delay between requests to comply with API rate limits

---

### 2. Symptom Extraction

**Purpose:** Extract structured symptom information from medical queries

```bash
python symptom_classification.py
```

**Functionality:**
- Utilizes 10 gold-standard annotated examples
- Extracts: symptom, body_location, duration, trigger
- Maps body locations to anatomical groups
- Calculates exact match accuracy and token-level F1 scores

**Estimated Runtime:** Approximately 15 seconds (10 questions)

**Expected output:**
```
{
  'symptom_accuracy': 0.90,
  'location_accuracy': 0.80,
  'duration_accuracy': 0.70,
  'trigger_accuracy': 0.60
}

f1_symptom      0.95
f1_location     0.87
f1_trigger      0.71
f1_duration     0.82
```

---

### 3. Retrieval-Augmented Generation (RAG)

**Purpose:** Build FAISS index and test semantic retrieval functionality

```bash
python RAG.py
```

**Functionality:**
- Loads 3,173 medical questions from `data/corpus.json`
- Generates embeddings using SentenceTransformers
- Builds FAISS L2-distance index
- Tests retrieval with sample query

**Estimated Runtime:** Approximately 30 seconds (one-time index building)

**Expected output:**
```
Batches: 100%|████████████████████| 100/100 [00:28<00:00,  3.52it/s]
Index size: 3173

RAG index built successfully
Index size: 3173

Test Question: Why does my chest feel tight when I wake up?

Retrieved:
- Why am I having pain in my chest?
- What are 5 causes of chest pain?
- Why does my breast hurt when I press it?
- Can anxiety cause chest pain?
- Why am I sweating at night while sleeping?

Answer:
Based on the provided context, chest tightness when waking up could be 
related to several factors. It might be associated with anxiety, acid reflux,
respiratory issues, or in rare cases, cardiac conditions...
```

---

### 4. Baseline Evaluation (Jupyter Notebook)

**Purpose:** Compare rule-based and machine learning baselines to LLM approach

```bash
# Launch Jupyter notebook
jupyter notebook baseline.ipynb
```

**Contents:**
- Rule-based classifier (keyword matching)
- TF-IDF + Random Forest
- TF-IDF + Logistic Regression
- Performance comparison tables

**Execution Instructions:**
1. Open `baseline.ipynb` in Jupyter interface
2. Select **Kernel** > **Restart & Run All**
3. Wait for completion (approximately 2 minutes)

**Expected results:**
- Rule-Based: Approximately 35% accuracy
- Random Forest: Approximately 49% accuracy
- Logistic Regression: Approximately 49% accuracy
- LLM (from LLM_intent.py): Approximately 77% accuracy

---

## Project Structure

```
MedQA/
├── app.py                          # Main Streamlit application
├── LLM_intent.py                   # Intent classification module
├── symptom_classification.py       # Symptom extraction module
├── RAG.py                          # Retrieval-augmented generation
├── baseline.ipynb                  # Baseline comparisons
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (API keys)
├── .gitignore                     # Git ignore rules
├── pyproject.toml                 # Project metadata
├── README.md                      # Project documentation
│
├── data/
│   └── corpus.json                # 3,173 medical questions for RAG
│
└── labels.json                    # 148 labeled questions for evaluation
```

---

## Data Files

### 1. labels.json

**Purpose:** Manually annotated evaluation dataset

**Structure:**
```json
{
  "id": 1,
  "question": "Are benign brain tumors serious?",
  "intent": "severity_question",
  "symptoms": ["benign brain tumors"],
  "duration": null,
  "trigger": null,
  "body_location": "neurological",
  "urgency": "high"
}
```

**Statistics:**
- **Total questions:** 148
- **Intent categories:** 18 (including treatment, severity, symptom, disease information, etc.)
- **Largest category:** treatment_question (62 questions, 42%)
- **Smallest category:** Various rare categories (2-6 questions each)

**Usage:**
- `LLM_intent.py` - Intent classification evaluation
- `symptom_classification.py` - Symptom extraction gold standard
- `baseline.ipynb` - Baseline comparison

---

### 2. data/corpus.json

**Purpose:** Retrieval corpus for RAG system

**Structure:**
```json
{
  "question": "Are benign brain tumors serious?"
}
```

**Statistics:**
- **Total questions:** 3,173
- **Source:** HealthSearchQA dataset
- **Content:** Real user health queries
- **Format:** JSON array of question objects

**Usage:**
- `RAG.py` - Semantic search and answer generation
- `app.py` - Retrieval context in main application

**File Size:** Approximately 215 KB (text-based)

---

## Troubleshooting

### Issue 1: API Key Not Found

**Error Message:**
```
ValueError: GROQ_API_KEY not found in environment variables
```

**Resolution:**
1. Verify `.env` file exists in project root directory
2. Confirm `.env` contains: `GROQ_API_KEY=your_key_here`
3. Ensure no whitespace characters around the equals sign
4. Restart Python interpreter or terminal session after creating `.env`

**Verification Test:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

---

### Issue 2: Module Import Errors

**Error Message:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Resolution:**
1. Activate virtual environment
2. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

**Verify Installation:**
```bash
pip list | grep streamlit
pip list | grep groq
pip list | grep faiss
```

---

### Issue 3: FAISS Installation Issues on Windows

**Error Message:**
```
ERROR: Could not find a version that satisfies the requirement faiss-cpu
```

**Resolution:**

**Option 1: Use conda (recommended for Windows)**
```bash
conda install -c conda-forge faiss-cpu
```

**Option 2: Install from conda-forge then pip**
```bash
conda install -c conda-forge faiss
pip install -r requirements.txt
```

**Option 3: Use faiss-cpu from PyPI**
```bash
pip install faiss-cpu --no-cache-dir
```

---

### Issue 4: Groq API Rate Limits

**Error Message:**
```
groq.RateLimitError: Rate limit exceeded
```

**Resolution:**
- Free tier has rate limits (typically 10-20 requests per minute)
- Script includes 0.15-second delays between requests
- If error persists, wait 60 seconds and retry
- For higher limits, upgrade Groq account tier

**Check Rate Limits:**
- Visit [console.groq.com](https://console.groq.com)
- Navigate to "Usage" or "Billing" section

---

### Issue 5: FAISS Index Building Performance

**Symptom:** `RAG.py` takes longer than 1 minute to build index

**Expected Behavior:**
- 3,173 questions with embedding generation requires approximately 30 seconds on standard hardware
- Progress bar indicates normal operation

**Optimization Options:**
```python
# Use batch_size parameter (already implemented)
corpus_embeds = embedder.encode(
    corpus_texts,
    batch_size=32,  # Adjust based on available RAM
    show_progress_bar=True
)
```

**For Large Corpora (>100K documents):**
```python
# Use IndexIVFFlat instead of IndexFlatL2
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(corpus_embeds)
index.add(corpus_embeds)
```

---

### Issue 6: Streamlit Port Already in Use

**Error Message:**
```
OSError: [Errno 98] Address already in use
```

**Resolution:**

**Option 1: Terminate existing Streamlit process**
```bash
# Linux/macOS
pkill -f streamlit

# Windows
taskkill /F /IM streamlit.exe
```

**Option 2: Specify alternative port**
```bash
streamlit run app.py --server.port 8502
```

**Option 3: Identify and terminate process on port 8501**
```bash
# Linux/macOS
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

---

### Issue 7: Memory Constraints with Large Models

**Symptom:** System encounters out-of-memory errors during embedding generation

**Resolution Options:**

**1. Reduce batch size:**
```python
# In RAG.py, modify:
corpus_embeds = embedder.encode(
    corpus_texts,
    batch_size=16,  # Reduce from default 32
    show_progress_bar=True
)
```

**2. Use lighter embedding model:**
```python
# In RAG.py, replace model:
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384 dimensions
# With:
embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")  # 384 dimensions, faster
```

**3. Process corpus in chunks:**
```python
# For very large corpora
chunk_size = 1000
all_embeds = []
for i in range(0, len(corpus_texts), chunk_size):
    chunk = corpus_texts[i:i+chunk_size]
    embeds = embedder.encode(chunk)
    all_embeds.append(embeds)
corpus_embeds = np.vstack(all_embeds)
```

---

### Issue 8: JSON Parsing Errors from LLM

**Error Message:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:** LLM occasionally returns text before or after JSON response

**Resolution:** Error handling is already implemented in code with try-except blocks

**If error persists, implement robust parsing:**
```python
import re

def extract_json(text):
    # Find JSON object in text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        return json.loads(match.group())
    return None
```

---

## Additional Notes

### Performance Benchmarks

**Test Hardware:** Intel i7 processor, 16GB RAM, No GPU acceleration

| Component | Runtime | Notes |
|-----------|---------|-------|
| FAISS Index Build | ~30 seconds | One-time operation |
| Intent Classification (30 queries) | ~45 seconds | ~1.5 seconds per query |
| Symptom Extraction (10 queries) | ~15 seconds | ~1.5 seconds per query |
| RAG Answer Generation | 3-4 seconds | Per query |
| Baseline Training | ~2 minutes | All 3 baselines combined |

---

### API Usage and Cost Structure

**Groq Free Tier Specifications:**
- **Rate Limits:** Approximately 10-20 requests per minute
- **Models:** Complete access to Llama-3.3-70B and Llama-3.1-8B
- **Cost:** No charge for research and development purposes
- **Query Limits:** No monthly cap announced (as of December 2024)

**Estimated Usage Per Query:**
- Intent classification: 1 API request per query
- Symptom extraction: 1 API request per symptom-related query
- RAG generation: 1 API request per answer
- **Total:** 2-3 API requests per complete pipeline execution

---

### System Extension Guidelines

#### Adding New Intent Categories

1. **Update labels.json:**
```json
{
  "id": 149,
  "question": "Your new question",
  "intent": "new_category_name",
  ...
}
```

2. **No code modifications required** - Zero-shot classification automatically incorporates new labels

3. **Execute test:**
```bash
python LLM_intent.py
```

---

#### Modifying LLM Models

**For Intent Classification (in LLM_intent.py):**
```python
# Line 59: Modify model parameter
model="llama-3.3-70b-versatile"  # Current configuration
# Alternative options:
model="llama-3.1-70b-versatile"  # Previous flagship model
```

**For Symptom Extraction and RAG (in symptom_classification.py and RAG.py):**
```python
# Line 104 and 64 respectively
model="llama-3.1-8b-instant"  # Current configuration
# Alternative options:
model="llama-3.3-70b-versatile"  # Higher capability
model="llama-3.1-70b-versatile"  # Alternative flagship
```

**Available Groq Models:**
- `llama-3.3-70b-versatile` - Highest capability
- `llama-3.1-70b-versatile` - Previous generation flagship
- `llama-3.1-8b-instant` - Optimized for speed and efficiency
- `mixtral-8x7b-32768` - Extended context length

---

#### Changing Embedding Model

**In RAG.py:**
```python
# Line 17: Modify embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Alternative options:
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")  # Enhanced quality
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual support
```

**Important:** After modifying embedding model, rebuild FAISS index by re-executing `RAG.py`

---

#### Replacing the Corpus

1. **Prepare new corpus in required format:**
```json
[
  {"question": "Your question 1"},
  {"question": "Your question 2"},
  ...
]
```

2. **Replace `data/corpus.json` with new file**

3. **Rebuild index:**
```bash
python RAG.py
```

**Corpus Size Recommendations:**
- Minimum: 100 questions (basic functionality)
- Optimal: 1,000-10,000 questions
- Maximum: 100,000+ questions (requires IndexIVFFlat)

---

### Memory Requirements by Corpus Size

| Corpus Size | RAM Required | Index Type | Build Time |
|-------------|--------------|------------|------------|
| 1,000 | 2 GB | IndexFlatL2 | ~10 seconds |
| 10,000 | 4 GB | IndexFlatL2 | ~30 seconds |
| 100,000 | 8 GB | IndexIVFFlat | ~5 minutes |
| 1,000,000 | 16+ GB | IndexIVFFlat + GPU | ~30 minutes |

---

### Production Deployment Considerations

**For production deployment, implement the following:**

1. **API Key Security:**
   - Utilize environment variables instead of `.env` files
   - Implement regular key rotation policies
   - Use enterprise secrets management systems (AWS Secrets Manager, HashiCorp Vault, etc.)

2. **Rate Limiting:**
   - Implement request queuing mechanisms
   - Add caching layer for frequent queries
   - Utilize exponential backoff for retry logic

3. **Error Handling:**
   - Implement comprehensive error logging with contextual information
   - Develop fallback response mechanisms
   - Monitor API usage and associated costs

4. **Performance Optimization:**
   - Pre-build FAISS index (avoid rebuilding on each startup)
   - Persist index to disk: `faiss.write_index(index, "faiss_index.bin")`
   - Load index from disk: `index = faiss.read_index("faiss_index.bin")`
   - Implement Redis or Memcached for query caching

5. **Monitoring and Observability:**
   - Track response time metrics
   - Monitor API error rates
   - Configure alerts for high latency or failures

**Example: Persistent FAISS Index Implementation**

```python
import os
import faiss

index_path = "faiss_index.bin"

# Save index after building
if not os.path.exists(index_path):
    # Build index (existing code)
    ...
    faiss.write_index(index, index_path)
else:
    # Load existing index
    index = faiss.read_index(index_path)
```

---

### Setup Verification Script

**Quick verification procedure:**

```bash
# Create test.py
cat > test.py << 'EOF'
from dotenv import load_dotenv
import os

load_dotenv()

print("Environment loaded successfully")

# Test API key
api_key = os.getenv("GROQ_API_KEY")
if api_key and api_key.startswith("gsk_"):
    print("Groq API key detected")
else:
    print("ERROR: Groq API key missing or invalid")

# Test imports
try:
    import streamlit
    print("Streamlit imported successfully")
except:
    print("ERROR: Streamlit import failed")

try:
    import groq
    print("Groq imported successfully")
except:
    print("ERROR: Groq import failed")

try:
    import faiss
    print("FAISS imported successfully")
except:
    print("ERROR: FAISS import failed")

try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformers imported successfully")
except:
    print("ERROR: SentenceTransformers import failed")

# Test data files
import os
if os.path.exists("labels.json"):
    print("labels.json located")
else:
    print("ERROR: labels.json not found")

if os.path.exists("data/corpus.json"):
    print("corpus.json located")
else:
    print("ERROR: corpus.json not found")

print("\nSetup verification complete")
EOF

# Execute test
python test.py
```

**Expected output:**
```
Environment loaded successfully
Groq API key detected
Streamlit imported successfully
Groq imported successfully
FAISS imported successfully
SentenceTransformers imported successfully
labels.json located
corpus.json located

Setup verification complete
```

---

## Learning Resources

### Understanding the Pipeline Components

1. **Intent Classification:**
   - Zero-shot Learning with Large Language Models: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
   - Groq Official Documentation: [https://console.groq.com/docs](https://console.groq.com/docs)

2. **Symptom Extraction:**
   - Few-shot Prompting Techniques: [https://www.promptingguide.ai/techniques/fewshot](https://www.promptingguide.ai/techniques/fewshot)
   - Clinical NLP Fundamentals: [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8861690/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8861690/)

3. **RAG (Retrieval-Augmented Generation):**
   - Original RAG Paper: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
   - FAISS Documentation: [https://github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki)
   - Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)

### Related Technologies and Tools

- **Vector Databases:** Pinecone, Weaviate, ChromaDB
- **LLM Frameworks:** LangChain, LlamaIndex
- **Alternative LLM APIs:** OpenAI, Anthropic (Claude), Cohere

---

## Contributing

For contributions to this project:

1. Review existing issues on the GitHub repository
2. Create detailed bug reports including:
   - Python version
   - Operating system
   - Complete error messages
   - Steps to reproduce the issue
3. Submit pull requests with comprehensive descriptions

---

## License

This project is developed for academic purposes. Refer to the repository for detailed license information.

---

## Support

For questions or technical issues:
- **GitHub Issues:** [https://github.com/shr7q/MedQA/issues](https://github.com/shr7q/MedQA/issues)
- **Project Authors:** Refer to README.md for contact information

---

## Quick Start Checklist

- [ ] Python 3.8 or higher installed
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Groq API account created
- [ ] API key configured in `.env` file
- [ ] `.env` file placed in project root directory
- [ ] Test script executed successfully
- [ ] Streamlit application launches without errors
- [ ] Sample query returns expected results

**Estimated setup time:** 15-20 minutes

---

**Last Updated:** December 2024  
**Project Version:** 0.1.0  
**Python Version:** 3.8 or higher

---

**To begin, execute:**
```bash
streamlit run app.py
```
