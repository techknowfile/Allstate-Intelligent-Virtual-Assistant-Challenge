#!python3
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, nps_chat
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import random
import pickle
import time
import textwrap

# SETTINGS
slowResponse = True # Adds a delays before bot responds

kbArticleID = -1

####################################
# Classifiers
####################################
# TODO: Put class in different file or pickle.
# Class is requiredd to load 'myYesNoAnswerClassificationObject' and 'myGreetingClassificationObject' -
#  instances of featureClassificationObject
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
classificationObjectInstanceFile.close();
classificationObjectInstanceFile = open('picklejar/myGreetingClassificationObject.pickle', 'rb')
myGreetingClassificationObject = pickle.load(classificationObjectInstanceFile)

# CONSTANTS
BOTNAME = "Red Queen (TSR)"
DEFAULT_USERNAME = "User"
NAME_COLUMN_WIDTH = 20

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
###################################################
# SECTION START:
#
# This section has been deprecated. Now using Brain
# stub.
####################################################
# class KBArticle:
# 	def __init__(self, resolution):
# 		self.resolution = resolution; # Tuple of resolution steps

# kbKeywordDict = {'address':(9, 'KB00206580'), 'login':(7, 'KB0083060')}

# kb1 = KBArticle(('Resolve the incident by unlocking the email account.',
# 'Verify with the user that their issue has been resolved.',
# '''Help the user understand and correct the root cause if necessary.
#    a. If the user has a mobile device connected to their work email, verify that the user has updated the password on their mobile device as well, since that may be the root cause of the account becoming locked.'''
#    ))

# kb2 = KBArticle(('Verify the customer\'s current billing address',
# 				'Verify the customer\'s current mailing address',
# 				'Verify that the customer is requesting a change to their billing address only or also their mailing address',
# 				'Update their billing address',
# 				'If the customer would like to update their mailing address as well, update their mailing address',
# 				'Confirm the customer\'s new billing address and, if applicable, their mailing address after the change'
# ))

# kbLibrary = {'KB0083060':kb1, 'KB00206580':kb2}

# ##################################################
# # END
# ##################################################

###################################################
## TEMP Class and instance ()
# TODO: This should come pickled from the KB with TFIDF matrix and /cosine similarity function (getKBKey(user_input_features))
class Brain:
	def __init__(self, tfs, brainEntryLookupDict):
		self.tfs = tfs
		self.brainEntryLookupDict = brainEntryLookupDict
	def getKBKey(self, feature_set):
		return 'KB00206580'
	def assistUser(self, brainEntry):
		for step in brainEntry.steps:
			if step[1] == 'add_domain_knowledge':
				tellUser(step[0])
				brainEntry.domainKnowledgeDict[step[2]].append(tellBot())
			if step[1] == 'confirm':
				for key, value in brainEntry.domainKnowledgeDict.items():
					if len(value) == 1:
						tellUser("Your {} has been updated to {}".format(key, value[0]))
					elif len(value) == 2:
						tellUser("Your {} has been updated from {} to {}".format(key, value[0], value[1]))


		
class BrainEntry:
	def __init__(self, domainKnowledgeDict, boolKnowledgeDict, steps):
		self.domainKnowledgeDict = domainKnowledgeDict
		self.steps = steps


myBrainEntry = BrainEntry(
	{'billing_address':[], 'mailing_address':[]},
	{'or_bool':None},
	[
		['What is your current billing address?', 'add_domain_knowledge', 'billing_address'],
		['What is your current mailing address?', 'add_domain_knowledge', 'mailing_address'],
		['Are you requesting to change your billing address only or also your mailing address?', 'or_bool'],
		['What would you like your new billing address to be?', 'update_domain_knowledge', 'billing_address'],
		['What would you like your new mailing address to be?', 'conditional_update_domain_knowledge', 'mailing_address'],
		['', 'confirm']
	]
	)
myBrain = Brain(None, {'KB00206580':myBrainEntry})
######################################################
# STUBS: END
######################################################

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
		kbKey = determineIssue(isFirstIssue) #Gets Key
		if kbKey:
			# get the Article from the Knowledge Base Library
			currentBrainEntry = myBrain.brainEntryLookupDict.get(kbKey)
			myBrain.assistUser(currentBrainEntry)

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
	# If user input is a greeting such as "Hi :)", wait for another input
	if isGreeting(userInput) and isFirstIssue:
		tellUser("Hi :)")
		userInput = tellBot()

	keywords = parseInput(userInput) # parse keywords from user input
	
	kbKey = myBrain.getKBKey(keywords) # TODO: John, hook up with actual feature set needed to perform cosine similarity

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