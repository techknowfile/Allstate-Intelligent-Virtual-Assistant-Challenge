import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize


class Brain:
	def __init__(self, kbWordsDict, brainEntryDict):
		self.kbWordsDict = kbWordsDict
		self.brainEntryDict = brainEntryDict

	#function takes the tokens and applies PorterStemmer
	def stemTokens(self, tokens):
		#stemmer object used to stem the tokens
		stemmer = PorterStemmer()
		stemmedTokens = []
		for token in tokens:
			stemmedTokens.append(stemmer.stem(token))
		return stemmedTokens

	#Takes the text and creates tokens
	def tokenize(self, text):
		tokens = nltk.word_tokenize(text)
		stems = self.stemTokens(tokens)
		return stems

	def getKBKey(self, userInput):
		self.kbWordsDict['input'] = 'update my address'

		#apply tfidf using the tokenize function made in line 24 and not including 'useless' words
		vectorizer = TfidfVectorizer(tokenizer=self.tokenize , stop_words='english', use_idf=True, ngram_range=(1, 3))
		tfidf = vectorizer.fit_transform(self.kbWordsDict.values())
		cosine_similarities = linear_kernel(tfidf[len(self.kbWordsDict)-1], tfidf).flatten()
		match = cosine_similarities.argsort()[:-3:-1]
		kbKey = list(self.kbWordsDict.keys())[match[1]]
		return kbKey

	def assistUser(self, brainEntry):
		for step in brainEntry.steps:
			if step[1] == 'add_domain_knowledge':
				tellUser(step[0]) # Ask user for domain knowledge
				brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)

			if step[1] == 'confirm':
				for key, value in brainEntry.domainKnowledgeDict.items():
					if len(value) == 1:
						tellUser("Your {} has been updated to {}".format(key, value[0]))
					elif len(value) == 2:
						tellUser("Your {} has been updated from {} to {}".format(key, value[0], value[1]))

			if step[1] == 'update_domain_knowledge':
				tellUser(step[0]) # Ask user for domain knowledge
				brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)

			if step[1] == 'conditional_update_domain_knowledge':
				# Check to see that the boolean flag has been set to True. If so, run conditional step...
				if boolKnowledgeDict.get('or_bool') == True:
					tellUser(step[0]) # Ask user for domain knowledge
					brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)


class BrainEntry:
	def __init__(self, domainKnowledgeDict, boolKnowledgeDict, steps):
		self.domainKnowledgeDict = domainKnowledgeDict
		self.boolKnowledgeDict = boolKnowledgeDict
		self.steps = steps