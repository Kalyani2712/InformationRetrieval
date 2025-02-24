#pip install torch
#pip install transformers
from transformers import pipeline
# Specify the summarization model explicitly
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# Example text
text = """
Quantum computing is one of the most exciting fields of modern technology. It aims to use the principles of quantum mechanics 
to perform calculations at speeds far beyond those of traditional computers. While classical computers rely on binary bits 
(0s and 1s), quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously, thanks to 
superposition. This property, combined with quantum entanglement, allows quantum computers to solve certain problems exponentially 
faster than their classical counterparts. However, the development of quantum computing is still in its early stages, and significant 
technical challenges remain, such as error correction and qubit coherence. Despite these hurdles, the potential applications of 
quantum computing in areas like cryptography, drug discovery, and optimization are vast, and researchers are optimistic about its future.
"""
# Summarization
summary = summarizer(text, max_length=100, min_length=50, do_sample=False)

# Output the summary
print("Summarization Model Used:", summarizer.model.name_or_path)
print("\nSummary of the Text:")
print(summary[0]['summary_text'])
