import nltk
from nltk import word_tokenize
from nltk.util import ngrams
text ="this is a sample text for unigram,bigram and trigram extraction using NLTK"
print(text)
token = word_tokenize(text.lower()) #tokenize text
unigrams = list(ngrams(token ,1))
print("\n Unigrams :", unigrams)
bigrams = list(ngrams(token ,2))
print("\n Bigrams :", bigrams)
trigrams = list(ngrams(token ,3))
print("\n Trigrams :", trigrams)


