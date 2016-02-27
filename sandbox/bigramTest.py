from nltk.util import ngrams

def getBigrams(userInput):
	n = 2
	bigrams = []
	bigramGenerator = ngrams(userInput.split(), n)
	for bigram in bigramGenerator:
		bigrams.append(" ".join(bigram))
	return bigrams
