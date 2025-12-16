from groq import Groq
from dotenv import load_dotenv
import os
import json
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load environment variables 
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


# Initialize Sentence Transformer and FAISS index
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Load corpus of medical questions (3172 questions)
with open("data/corpus.json", "r") as f:
    corpus = json.load(f)


# Create embeddings and build FAISS index
corpus_texts = [item["question"] for item in corpus]
corpus_embeds = embedder.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=True)
corpus_embeds = corpus_embeds.astype("float32")


# Build FAISS index
dimension = corpus_embeds.shape[1]  
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeds)

# Print index size
print("Index size:", index.ntotal)


# Retrieve similar questions using FAISS 
def retrieve_similar(query, k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [corpus_texts[i] for i in indices[0]]


# RAG Answer Generation
def rag_answer(question, k=5):
    retrieved = retrieve_similar(question, k)
    
    context = "\n".join([f"- {doc}" for doc in retrieved])
    
    prompt = f"""
    You are designed to answer strictly medical related queries. Anything that is not medical related 
    should be responded with "Sorry I can only answer medical related queries.".

    Use the retrieved similar questions below to provide a medically safe and helpful answer.
    If context is insufficient, say "more information is needed".

    Retrieved similar questions:
    {context}

    User question: {question}

    Provide your best answer:
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    msg = response.choices[0].message.content
    if isinstance(msg, list):
        msg = "".join([p.text for p in msg])

    return msg, retrieved


if __name__ == "__main__":
    print("RAG index built successfully")
    print("Index size:", index.ntotal)

    test_q = "Why does my chest feel tight when I wake up?"
    answer, retrieved = rag_answer(test_q)

    print("\nTest Question:", test_q)
    print("\nRetrieved:")
    for r in retrieved:
        print("-", r)

    print("\nAnswer:\n", answer)
