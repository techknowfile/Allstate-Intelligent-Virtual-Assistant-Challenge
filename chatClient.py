#!python3
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, nps_chat
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer

import random
import pickle
import time
import textwrap

# SETTINGS
slowResponse = False # Adds a delays before bot responds

kbArticleID = -1

###################################################################################################
#                                         CLASS DEFINITIONS
###################################################################################################
# TODO: These two classes currently must be copied/pasted across to kbTrainer (or vice-verse)
#       when a change is made to the class... because I suck at pickling. And if I use a
#       separate class file the brain isn't able to access the functions in this chatClient class
#       (trying to import chatClient functions into a brain.py class breaks everything)
###################################################################################################

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
		self.kbWordsDict['input'] = userInput

		#apply tfidf using the tokenize function made in line 24 and not including 'useless' words
		vectorizer = TfidfVectorizer(tokenizer=self.tokenize , stop_words='english', use_idf=True, ngram_range=(1, 3))
		tfidf = vectorizer.fit_transform(self.kbWordsDict.values())
		cosine_similarities = linear_kernel(tfidf[len(self.kbWordsDict)-1], tfidf).flatten()
		match = cosine_similarities.argsort()[:-3:-1]
		score = cosine_similarities[match[1]]
		if score > 0:
			kbKey = list(self.kbWordsDict.keys())[match[1]]
			print(kbKey)
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
###################################################################################################
# TODO: Put class in different file or pickle.
###################################################################################################
# Class is requiredd to load 'myYesNoAnswerClassificationObject' and 'myGreetingClassificationObject'
# instances of featureClassificationObject
###################################################################################################
class featureClassificationObject:
    def __init__(self, word_features, classifier):
        self.word_features = word_features
        self.classifier = classifier
    def find_features(self, post):
        words = set(post)
        features = {}
        for w in self.word_features:
            if len(w) > 1:
                features[w] = (w in words)
        return features



###################################################################################################
###################################################################################################
#                                          HER BRAIN
###################################################################################################
brain_file = open('picklejar/brain.pickle', 'rb')
brain = pickle.load(brain_file)
brain_file.close()
###################################################################################################


###################################################################################################
#                                        Classifiers
###################################################################################################
# Load yesNoAnswerClassification instance
# myYesNoAnswerClassificationObject: contains the word_features, trained NB classifier, and 'find_features([])' func
classificationObjectInstanceFile = open('picklejar/myYesNoAnswerClassificationObject.pickle', 'rb')
myYesNoAnswerClassificationObject = pickle.load(classificationObjectInstanceFile)
classificationObjectInstanceFile.close();
classificationObjectInstanceFile = open('picklejar/myGreetingClassificationObject.pickle', 'rb')
myGreetingClassificationObject = pickle.load(classificationObjectInstanceFile)
classificationObjectInstanceFile.close();

###################################################################################################
#                                         CONSTANTS
###################################################################################################
BOTNAME = "Red Queen (TSR)"
DEFAULT_USERNAME = "User"
NAME_COLUMN_WIDTH = 20

###################################################################################################
#                                         User data
###################################################################################################
username = ''

###################################################################################################
#                                        NLTK Stuff
###################################################################################################
# NLTK Objects
lemmatizer = WordNetLemmatizer()

# NLTK sets and lists
stop_words = set(stopwords.words('english'))
stop_words.remove('no')

###################################################################################################
# ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
###################################################################################################
#                                           MAIN CODE
###################################################################################################
# ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~
###################################################################################################
def main():
	''' TODO: Docstring'''

	allIssuesResolved = False

	clearScreen();
	getUsername();
	kbArticle = None
	isFirstIssue = True

	while not allIssuesResolved:
		kbKey = determineIssue(isFirstIssue) #Gets Key
		if kbKey:
			tellUser('I\'d be happy to help you with that ^_^')
			# get the Article from the Knowledge Base Library
			currentBrainEntry = brain.brainEntryDict.get(kbKey)

			################################################
			#**************    RESOLUTION    ***************
			################################################
			brain.assistUser(currentBrainEntry)

		# Check if user says no, yes, or gives another problem
		needsMoreHelp = yesNoQuestion('Is there anything else I can help you with today?')
		if needsMoreHelp == 'Yes':
			allIssuesResolved = False

			# TODO: Should add something that determines if they gave the next issue immediately instead of just giving a 'yes' answer
		else:
			allIssuesResolved = True
		isFirstIssue = False # Set flag to false so that different prompt is provided by bot for next issue

	tellUser("I'm glad I could help. Have a great day!")


###################################################################################################
#                                             FUNCTIONS
###################################################################################################
def determineIssue(isFirstIssue):
	if isFirstIssue:
		tellUser("Hello, {}! How can I help you today? ".format(username))
	else:
		tellUser("What else can I assist you with?")
	userInput = tellBot() # Get user input
	# If user input is a greeting such as "Hi :)", wait for another input
	if isGreeting(userInput) and isFirstIssue:
		tellUser("Hi :)")
		userInput = tellBot()

	# keywords = parseInput(userInput) # parse keywords from user input
	
	kbKey = brain.getKBKey(userInput) # TODO: John, hook up with actual feature set needed to perform cosine similarity

	# Deprecated from initial stub
	# matched_keywords = [(keyword, kbKeywordDict.get(keyword)) for keyword in keywords if keyword in kbKeywordDict]
	# matched_keywords = sorted(matched_keywords, key=lambda keyword: keyword[1][0], reverse=True)
	
	if kbKey:
		# keyword matched!
		issueDetermined = True

		# DEPRECATED: Returns only the highest rated match. Remove [0] to return all matched Keyword/KB article pairs
		# return matched_keywords[0][1]

		# Return the key to the KB article
		return kbKey
	
	else:
		tellUser("I'm sorry, I'm not able to assist with that issue.")


def tellUser(response):
	# Using sys.stdout.write in order to update the text that has already been printed out in the console.
	# This can be used for a multi-threaded implementation of the app.
	# TODO: Test on linux/OSX.
	# curses module would be a better option; however, it doesn't support Windows.

	sys.stdout.write("\r%20s" % username + ": ") #display the user input line in the console.
	if slowResponse == True:
		time.sleep(2)
	sys.stdout.write("\r" + textwrap.fill("{:>20}: {:<60}\n".format(BOTNAME, response), 70, subsequent_indent="                      ") + "\n") # Replace the "user input line" with the bots' response

	# print(BOTNAME + ":", response)

def tellBot():
	return input("%20s: "%(username))

def clearScreen():
	''' Function to clear current screen based on current OS '''
	os.system('cls' if os.name == 'nt' else 'clear')

def parseInput(userInput):
	# Tokenize user input into a list of words.
	words = word_tokenize(userInput)
	# Remove stop words and words less than 2 characters long, then LEMMATIZE input
	# words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words and len(w) > 1] 

	parsed_input = [w.lower() for w in words if w not in stop_words and len(w) > 1]
	return parsed_input

def getUsername():
	global username
	username = input('''Welcome to TeamDudley\'s Virtual Assistant!

To get started, simply enter your name: ''')
	if username == '':
		username = DEFAULT_USERNAME
	clearScreen();

def yesNoQuestion(question):
	tellUser(question)
	userInput = tellBot()
	parsed_input = parseInput(userInput)
	feature_set = myYesNoAnswerClassificationObject.find_features(parsed_input)
	return(myYesNoAnswerClassificationObject.classifier.classify(feature_set))

def isGreeting(userInput):
	parsed_input = parseInput(userInput)
	feature_set = myGreetingClassificationObject.find_features(parsed_input)
	return (True if myGreetingClassificationObject.classifier.classify(feature_set) == 'Greeting' else False)





# END OF DOCUMENT
# Load all functions, then run the main function (removes need for forward declarations,
# which don't exist in Python)
if __name__=="__main__":
   main()