#!python3
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, nps_chat
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import random
import pickle

kbArticleID = -1

####################################
# Classifiers
####################################
# TODO: Put class in different file or pickle.
# Class is requiredd to load 'myYesNoAnswerClassificationObject', and instance of featureClassificationObject
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
# Load yesNoAnswerClassification instance
# myYesNoAnswerClassificationObject: contains the word_features, trained NB classifier, and 'find_features([])' func
classificationObjectInstanceFile = open('picklejar/myYesNoAnswerClassificationObject.pickle', 'rb')
myYesNoAnswerClassificationObject = pickle.load(classificationObjectInstanceFile)



# CONSTANTS
BOTNAME = "Alice (TSR)"
DEFAULT_USERNAME = "User"

# User data
username = ''

# NLTK Objects
lemmatizer = WordNetLemmatizer()

# NLTK sets and lists
stop_words = set(stopwords.words('english'))
stop_words.remove('no')

#################################################
# STUBS | Test data | Test Classes | Test Objects
#################################################
class KBArticle:
	def __init__(self, resolution):
		self.resolution = resolution; # Tuple of resolution steps

kbKeywordDict = {'address':(9, 'KB00206580'), 'login':(7, 'KB0083060')}

kb1 = KBArticle(('Resolve the incident by unlocking the email account.',
'Verify with the user that their issue has been resolved.',
'''Help the user understand and correct the root cause if necessary.
   a. If the user has a mobile device connected to their work email, verify that the user has updated the password on their mobile device as well, since that may be the root cause of the account becoming locked.'''
   ))

kb2 = KBArticle(('Verify the customer\'s current billing address',
				'Verify the customer\'s current mailing address',
				'Verify that the customer is requesting a change to their billing address only or also their mailing address',
				'Update their billing address',
				'If the customer would like to update their mailing address as well, update their mailing address',
				'Confirm the customer\'s new billing address and, if applicable, their mailing address after the change'
))

kbLibrary = {'KB0083060':kb1, 'KB00206580':kb2}

##############################################
# Start of main code
##############################################
def main():
	''' TODO: Docstring'''

	allIssuesResolved = False
	clearScreen();
	getUsername();
	kbArticle = None
	isFirstIssue = True

	while not allIssuesResolved:
		kwArticlePair = determineIssue(isFirstIssue) #Gets Key
		if kwArticlePair:
			# get the Article from the Knowledge Base Library
			kbArticle = kbLibrary.get(kwArticlePair[1])
			for step in kbArticle.resolution:
				tellUser(step)

		# Check if user says no, yes, or gives another problem
		needsMoreHelp = yesNoQuestion('Is there anything else I can help you with today?')
		if needsMoreHelp == 'Yes':
			allIssuesResolved = False

			# TODO: Should add something that determines if they gave the next issue immediately instead of just giving a 'yes' answer
		else:
			allIssuesResolved = True
		isFirstIssue = False # Set flag to false so that different prompt is provided by bot for next issue

	tellUser("I'm glad I could help. Have a great day!")

	

def determineIssue(isFirstIssue):
	if isFirstIssue:
		tellUser("Hello, {}! How can I help you today? ".format(username))
	else:
		tellUser("What else can I assist you with?")
	userInput = tellBot() # Get user input
	keywords = parseInput(userInput) # parse keywords from user input
	
	matched_keywords = [(keyword, kbKeywordDict.get(keyword)) for keyword in keywords if keyword in kbKeywordDict]
	matched_keywords = sorted(matched_keywords, key=lambda keyword: keyword[1][0], reverse=True)
	
	if matched_keywords:
		# keyword matched!
		issueDetermined = True
		# Returns only the highest rated match. Remove [0] to return all matched Keyword/KB article pairs
		return matched_keywords[0][1]
	
	else:
		tellUser("I'm sorry, I'm not able to assist with that issue.")


def tellUser(response):
	print(BOTNAME + ":", response)

def tellBot():
	return input(username + ": ")

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

def getYesNoAnswerClassifier():
	'''Loads pickled classifier for classifying if a feature set (user input) was a \'yes\' response or \'no\' response.'''
	f = open('picklejar/yesNoAnswerClassifier.pickle', 'rb')
	yesNoAnswerClassifier = pickle.load(f)
	f.close()
	return yesNoAnswerClassifier

def yesNoQuestion(question):
	tellUser(question)
	userInput = tellBot()
	parsed_input = parseInput(userInput)
	feature_set = myYesNoAnswerClassificationObject.find_features(parsed_input)
	return(myYesNoAnswerClassificationObject.classifier.classify(feature_set))

# END OF DOCUMENT
# Load all functions, then run the main function (removes need for forward declarations,
# which don't exist in Python)
if __name__=="__main__":
   main()