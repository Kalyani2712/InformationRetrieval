import nltk
import gensim
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
docs = [
    "Information retrieval is a key area in search engines.",
    "Machine learning helps improve search relevance.",
    "Deep learning and AI are advancing information retrieval.",
    "Search engines use algorithms to rank results.",
    "Artificial intelligence enhances text processing."
]
tokenized_docs = [word_tokenize(doc.lower()) for doc in docs]
model = gensim.models.Word2Vec(tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)
def get_vector(sentence):
    vectors = [model.wv[w] for w in word_tokenize(sentence.lower()) if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
def search(query):
    query_vec, doc_vecs = get_vector(query), np.array([get_vector(doc) for doc in docs])
    scores = cosine_similarity([query_vec], doc_vecs)[0]
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nSearch Results for: {query}")
    for doc, score in ranked:
        print(f"Score: {score:.4f} | {doc}" if score > 0 else "No relevant results.")
search(input("Enter search query: "))
