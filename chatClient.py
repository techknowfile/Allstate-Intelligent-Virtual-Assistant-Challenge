#!python3
import os
import sys
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, nps_chat
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams

import random
import pickle
import time
import textwrap

import subprocess

from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from termcolor import cprint, colored
from pyfiglet import figlet_format

###################################################################################################
#                                          SETTINGS
###################################################################################################
slowResponse = True # Adds a delays before bot responds
RESPONSE_TIME = 1.5
kbArticleID = -1
openingSequence = True

###################################################################################################
#                                         CONSTANTS
###################################################################################################
CHAT_CLIENT_NAME = "Smart Support"
BOTNAME = "Sarah Bellum"
DEFAULT_USERNAME = "User"
SYSTEM = "System"
NAME_COLUMN_WIDTH = 20
CONTACT_CUSTOMER_SUPPORT = "I'm so sorry, it appears that this issue may need to be resolved over the phone. Please contact our customer support at (555) 123-1337"
BOT_SOLVED_PROBLEM_LIKE_A_BAU5 = "Awesome!"
BORDER = "======================================================================"

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
		match = cosine_similarities.argsort()[:-3:-1] # Get the indexes of the top two matches
		# The first match is a sanity check... the input features should match themselves 100%. The second match is the KB article we're looking for.
		score = cosine_similarities[match[1]]
		if score > 0:
			kbKey = list(self.kbWordsDict.keys())[match[1]]
			return kbKey

	def assistUser(self, brainEntry):
		issueResolved = False
		resolveStepEncountered = False
		domainTypeUpdatedWithOnlyOneEntry = []
		for step in brainEntry.steps:

			######################################################
			#  >>       Asking user for domain knowledge
			######################################################
			if step[1] == 'add_domain_knowledge':
				tellUser(step[0]) # Ask user for domain knowledge
				try:
					brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)
				except:
					printError("{} not found in my domain knowledge.".format(step[2]))
					printSystem("Maybe I can help. Attempting to add {} to {}'s knowledge bank".format(step[2], BOTNAME))
					try:
						brainEntry.domainKnowledgeDict[step[2]] = []
						brainEntry.domainKnowledgeDict[step[2]].append(userInput)
						domainTypeUpdatedWithOnlyOneEntry.append(step[2])
						printSystem("Done. You should be alright now, {}.".format(BOTNAME))
						tellUser("Thank you, System! Sorry about that, {}. Now, where were we?".format(username))
						tellUser("Oh yes...")
					except:
						printError("{} still not found in my domain knowledge.".format(step[2]))
						printError("{} is likely to crash soon.".format(BOTNAME))
			######################################################
			# >>    Confirming the acquired domain knowledge
			######################################################
			elif step[1] == 'confirm':
				for key, value in brainEntry.domainKnowledgeDict.items():
					if len(value) == 1 and not domainTypeUpdatedWithOnlyOneEntry:
						# Do nothing? This currently only occurs if the user chose not to update the value, so it shouldn't be relevant.
						pass
						# tellUser("Your {} has been updated to {}".format(key, value[0]))
					elif len(value) > 1:
						if value[len(value)-2] != value[len(value)-1]: # make sure last two values aren't identical...
							tellUser("Your {} has been updated from {} to {}".format(key.replace("_", " "), value[len(value)-2], value[len(value)-1]))
						else:
							tellUser("Your {} has been updated to {}.".format(key.replace("_", " "), value[-1]))
					elif  len(value) == 1 and key in domainTypeUpdatedWithOnlyOneEntry:
						tellUser("Your {} has been updated to {}.".format(key.replace("_", " "), value[-1]))
			######################################################
			# >>                 Resolve issue
			######################################################
			elif step[1] == 'resolve':
				tellUser(step[0])
				resolveStepEncountered = True

			######################################################
			# >>                 Yes No Question
			######################################################
			elif step[1] == 'yes_no':
				response = yesNoQuestion(step[0]) # Ask the use a yes no Question
				response = (True if response[0] == 'Yes' else False)
				try:
					brainEntry.boolKnowledgeDict[step[2]] = response
				except:
					printError("{} not found in my domain knowledge.".format(step[2]))
					printSystem("Maybe I can help. Attempting to add {} to {}'s knowledge bank".format(step[2], BOTNAME))
					try:
						brainEntry.domainKnowledgeDict[step[2]] = []
						brainEntry.domainKnowledgeDict[step[2]].append(userInput)
						domainTypeUpdatedWithOnlyOneEntry.append(step[2])
						printSystem("Done. You should be alright now, {}.".format(BOTNAME))
						tellUser("Thank you, System! Sorry about that, {}. Now, where were we?".format(username))
						tellUser("Oh yes...")
					except:
						printError("{} still not found in my domain knowledge.".format(step[2]))
						printError("{} is likely to crash soon.".format(BOTNAME))
				step_index = brainEntry.steps.index(step)
				if step_index is len(brainEntry.steps)-1:
					if response:
						tellUser(BOT_SOLVED_PROBLEM_LIKE_A_BAU5)
					else:
						tellUser(CONTACT_CUSTOMER_SUPPORT)
				elif resolveStepEncountered == True:
					if response:
						tellUser(BOT_SOLVED_PROBLEM_LIKE_A_BAU5)
						break

			######################################################
			# >>         Updating domain knowledge
			######################################################
			elif step[1] == 'update_domain_knowledge':
				tellUser(step[0]) # Ask user for domain knowledge
				try:
					userInput = tellBot()
					brainEntry.domainKnowledgeDict[step[2]].append(userInput) # Append domain knowledge to list for that specific domain knowledge (retains values)
				except:
					printError("{} not found in my domain knowledge.".format(step[2]))
					printSystem("Maybe I can help. Attempting to add {} to {}'s knowledge bank".format(step[2], BOTNAME))
					try:
						brainEntry.domainKnowledgeDict[step[2]] = []
						brainEntry.domainKnowledgeDict[step[2]].append(userInput)
						domainTypeUpdatedWithOnlyOneEntry.append(step[2])
						printSystem("Done. You should be alright now, {}.".format(BOTNAME))
						tellUser("Thank you, System! Sorry about that, {}. Now, where were we?".format(username))
						tellUser("Oh yes...")
					except:
						printError("{} still not found in my domain knowledge.".format(step[2]))
						printError("{} is likely to crash soon.".format(BOTNAME))


			######################################################
			# >>   Conditionally updating domain knowledge
			######################################################
			elif step[1] == 'conditional_update_domain_knowledge':
				##############################################################
				#  The user has already seen a boolean step for determining if 
				#  they would like to update specific domain knowledge.
				#  Based on their choice in that step, do this step.
				##############################################################
				# Check to see that the boolean flag has been set to True. If so, run conditional step...
				try:
					if brainEntry.boolKnowledgeDict.get(step[3]) == True:
						tellUser(step[0]) # Ask user for domain knowledge
						brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)
				except:
					print("ERROR: Something appears to be wrong with a conditional_update_domain_knowledge step or the process for assisting the user with that step.\nIf you are seeing this, I have failed you. I apologize for the inconvenience.")
			
			######################################################
			# >>   Setting boolean domain knowledge
			# >>               OR-ALSO
			######################################################
			elif step[1] == 'or_also_bool':
				#######################################
				# Ask user 'x only or also y' question
				#######################################

				# Get bool keys. These will be referenced later on by a conditional_update_domain_knowledge step
				key0 = step[2][0][0]
				key1 = step[2][1][0]

				tellUser(step[0]) # Ask user "x only or also y" question.
				input = tellBot() # get response
				input_features = parseInput(input) # tokenize response
				doBoth = getOrAlsoChoice(input_features, step[2][0][1], step[2][1][1]) # process respone
				try:
					if doBoth:
						brainEntry.boolKnowledgeDict[key0] = True
						brainEntry.boolKnowledgeDict[key1] = True
					else:
						brainEntry.boolKnowledgeDict[key0] = True
						brainEntry.boolKnowledgeDict[key1] = False
				except:
					print("ERROR: {} or {} not found in boolKnowledgeDict".format(key0, key1))

			######################################################
			# >>   Setting boolean domain knowledge
			# >>      OR (supports one, the other, or both)
			######################################################
			elif step[1] == 'or_bool':
				#######################################
				# Ask user 'x or y' question
				#######################################

				# Get bool keys. These will be referenced later on by a conditional_update_domain_knowledge step
				key0 = step[2][0][0]
				key1 = step[2][1][0]

				tellUser(step[0]) # Ask user "x only or also y" question.
				input = tellBot() # get response
				input_features = parseInput(input) # tokenize response
				decision = getOrChoice(input_features, step[2][0][1], step[2][1][1]) # process respone
				try:
					if decision == 0: # User chose the first option
						brainEntry.boolKnowledgeDict[key0] = True
						brainEntry.boolKnowledgeDict[key1] = False
					elif decision == 1: # User chose the second option
						brainEntry.boolKnowledgeDict[key0] = False
						brainEntry.boolKnowledgeDict[key1] = True
					elif decision == 2: # User wants both
						brainEntry.boolKnowledgeDict[key0] = True
						brainEntry.boolKnowledgeDict[key1] = True
					else: # User is hacking
						tellUser("Nice try ^_0")


				except:
					print("ERROR: {} or {} not found in boolKnowledgeDict".format(key0, key1))

			######################################################
			# >>          COMPLICATED YES-NO ('Verify')
			# >>   Ask yes no question about subject we haven't
			# >>   created a domainKnowledgeDictionary entry on 
			# >>   yet, then ask another yes-no question if they
			# >>   say yes.
			######################################################
			elif step[1] == 'if_yes_no_yes_no':
				response = yesNoQuestion(step[0]) # Ask the use a yes no Question
				response = (True if response[0] == 'Yes' else False)
				brainEntry.boolKnowledgeDict[step[3]] = response
				if response:
					response = yesNoQuestion(step[2])
					response = (True if response[0] == 'Yes' else False)
					step_index = brainEntry.steps.index(step)
					if step_index is len(brainEntry.steps)-1:
							if response:
								tellUser(BOT_SOLVED_PROBLEM_LIKE_A_BAU5)
							else:
								tellUser(CONTACT_CUSTOMER_SUPPORT)
				else:
					step_index = brainEntry.steps.index(step)
					if step_index is len(brainEntry.steps)-1:
						tellUser(CONTACT_CUSTOMER_SUPPORT)
						break

			######################################################
			# >>          COMPLICATED YES-NO ('UPDATE')
			# >>   Ask yes no question about subject we haven't
			# >>   created a domainKnowledgeDictionary entry on 
			# >>   yet, then ask for and retrieve new domain 
			# >>   knowledge.
			######################################################
			elif step[1] == 'if_yes_no_conditional_update_domain_knowledge':
				response = yesNoQuestion(step[0]) # Ask the use a yes no Question
				response = (True if response[0] == 'Yes' else False)
				brainEntry.boolKnowledgeDict[step[3]] = response
				if response:
					try:
						tellUser("What would you like your new {} to be?".format(step[3].replace("_", " ")))
						brainEntry.domainKnowledgeDict[step[3]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)
						domainTypeUpdatedWithOnlyOneEntry.append(step[3])
					except Exception as e:
						print(e)
						print("Step: {}, DKD: {}".format(step, brainEntry.domainKnowledgeDict))
						print("ERROR: Something appears to be wrong with a 'if_yes_no_conditional_update_domain_knowledge' step or the process for assisting the user with that step.\nIf you are seeing this, I have failed you. I apologize for the inconvenience.")

					
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
try:
	brain_file = open('picklejar/brain.pickle', 'rb')
	brain = pickle.load(brain_file)
	brain_file.close()
except:
	brain = None
###################################################################################################


###################################################################################################
#                                        Classifiers
###################################################################################################
# Load yesNoAnswerClassification instance
# myYesNoAnswerClassificationObject: contains the word_features, trained NB classifier, and 'find_features([])' func
classificationObjectInstanceFile = open('picklejar/myYesNoAnswerClassificationObject.pickle', 'rb')
myYesNoAnswerClassificationObject = pickle.load(classificationObjectInstanceFile)
classificationObjectInstanceFile.close();

# Load greeting classification instance
# myGreetingClassificationObject: contains the word_features, trained NB classifier, and 'find_features([])' func
classificationObjectInstanceFile = open('picklejar/myGreetingClassificationObject.pickle', 'rb')
myGreetingClassificationObject = pickle.load(classificationObjectInstanceFile)
classificationObjectInstanceFile.close();

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
	clearScreen()
	printHeader()
	getUsername()
	if openingSequence:
		startingSequence()
	brainEntryFound = False # Deprecated TODO: Delete
	isFirstIssue = True
	newKBKey = None # Used to determine if the user has another issue and provides it with their 'yes' response, as opposed to just saying 'yes'
	while not allIssuesResolved:
		if not newKBKey:
			kbKey = determineIssue(isFirstIssue) #Gets Key
		else:
			kbKey = newKBKey
			newKBKey = None

		if kbKey:
			tellUser('I\'d be happy to help you with that ^_^')
			# get the Article from the Knowledge Base Library

			################################################
			#**************    RESOLUTION    ***************
			################################################

			currentBrainEntry = brain.brainEntryDict.get(kbKey)
			if  currentBrainEntry:
				brain.assistUser(currentBrainEntry)
			else:
				tellUser("Oh no, where is my mind?!")
				tellUser("I appear to have been lobotomized")
				tellUser("I understand what you need help with, but I cant seem to find the part of my brain needed :(")
				tellUser("How embarrassing...")	
		else:
			tellUser("I'm sorry, I'm not able to assist with that issue.")

		# Check if user says no, yes, or gives another problem
		needsMoreHelp = yesNoQuestion('Is there anything else I can help you with today?')
		if needsMoreHelp[0] == 'Yes':
			allIssuesResolved = False
			newKBKey = brain.getKBKey(needsMoreHelp[1])

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
	
	kbKey = brain.getKBKey(userInput)

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


def tellUser(response):
	# Using sys.stdout.write in order to update the text that has already been printed out in the console.
	# This can be used for a multi-threaded implementation of the app.
	# TODO: Test on linux/OSX.
	# curses module would be a better option; however, it doesn't support Windows.

	sys.stdout.write(colored("\r%20s" % username + ": ")) #display the user input line in the console.
	if slowResponse == True:
		time.sleep(RESPONSE_TIME)
	sys.stdout.write(colored("\r" + textwrap.fill("{:>20}: {:<60}\n".format(BOTNAME, response), 70, subsequent_indent="                      ") + "\n", 'cyan')) # Replace the "user input line" with the bots' response

	# print(BOTNAME + ":", response)

def tellBot():
	sys.stdout.write(colored("%20s: "%(username)))
	return input()

def clearScreen():
	''' Function to clear current screen based on current OS '''
	os.system('cls' if os.name == 'nt' else 'clear')

def parseInput(userInput):
	# Tokenize user input into a list of words.
	loweredInput = userInput.lower()
	text = re.sub(r'[\.,-\?\']', '', loweredInput)
	words = word_tokenize(text)
	# Remove stop words and words less than 2 characters long, then LEMMATIZE input
	# words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words and len(w) > 1] 
	parsed_input = [w for w in words if len(w) > 1] # removed 'if w not in stop_words', because we need some of those words... TODO?
	bigrams = getBigrams(text)
	parsed_input.extend(bigrams)
	return parsed_input

def getBigrams(userInput):
	n = 2
	bigrams = []
	bigramGenerator = ngrams(userInput.split(), n)
	for bigram in bigramGenerator:
		bigrams.append(" ".join(bigram))
	return bigrams

def getUsername():
	global username
	sys.stdout.write(colored("%20s: "% "Enter your name"))
	username = input()
	if username == '':
		username = DEFAULT_USERNAME
	elif len(username) > 19:
		username = username[:19]
	if openingSequence:
		delay_print(' Thank you, {}. One moment, please.'.format(username), 0.1)
	cprint('\n{}'.format(BORDER), 'white')

def yesNoQuestion(question):
	tellUser(question)
	userInput = tellBot()
	parsed_input = parseInput(userInput)
	feature_set = myYesNoAnswerClassificationObject.find_features(parsed_input)
	return((myYesNoAnswerClassificationObject.classifier.classify(feature_set), userInput))

def isGreeting(userInput):
	parsed_input = parseInput(userInput)
	feature_set = myGreetingClassificationObject.find_features(parsed_input)
	return (True if myGreetingClassificationObject.classifier.classify(feature_set) == 'Greeting' else False)

def getOrAlsoChoice(input_features, feature_set_0, feature_set_1):
	if any(x in input_features for x in ('only', 'just')): # these words indicate that the user only wants the first option, not both.
		return False
	for feature in input_features:
		both_features = ['both', 'yes', 'also', 'as well', 'sure', 'fine'] #if they any of these, then return "both"
		feature_ngram = feature_set_1 # if they say 'x' in 'or also x', return "both" (support single words and bigrams only.. )
		both_features.append(feature_ngram)
		if feature in both_features:
			return True

	# default to only the first option		
	return False

def getOrChoice(input_features, feature_set_0, feature_set_1):
	if any(x in input_features for x in ('only', 'just')): # these words indicate that the user only wants the first option, not both.
		if feature_set_0 in input_features:
			return 0
		elif feature_set_1 in input_features:
			return 1

	for feature in input_features:
		both_features = ['both', 'yes'] #if they any of these, then return "both"
		if feature in both_features:
			return 2

	if all(x in input_features for x in (feature_set_0, feature_set_1)):
		return 2

	elif feature_set_0 in input_features:
		return 0

	elif feature_set_1 in input_features:
		return 1

	# default to only the first option		
	return False

def printHeader():
	cprint(figlet_format(CHAT_CLIENT_NAME, font='small'), 'white')
	cprint(BORDER, 'white')

def startingSequence():
	delay_print(" . . . ", 0.5, 'Loading Brain')
	if brain:
		sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", "Loading Brain . . . COMPLETE"))
		sys.stdout.write(colored('%s\n' % sys_string, 'yellow'))
		sys.stdout.flush()
		time.sleep(0.5)
	else:
		sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", "Loading Brain . . . FAILED"))
		sys.stdout.write(colored('%s\n' % sys_string, 'yellow'))
		sys_string = "\r" + textwrap.fill("{:>20}: {}".format("ERROR", "Pickled brain not found in picklejar. Run brainBuilder.py, then try again."), 70, subsequent_indent="                      ")
		sys.stdout.write(colored('%s\n' % sys_string, 'red'))
		sys.exit(0)

	delay_print(" . . . . ", 0.5, 'Starting AI')
	sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", "Starting AI . . . . ONLINE"))
	sys.stdout.write(colored('%s\n' % sys_string, 'yellow'))
	sys.stdout.flush()
	cprint(BORDER, 'white')
def delay_print(s, delay, message=''):
	# Use only for startingSequence
	sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", message), 70, subsequent_indent="                      ")
	for c in s:
		sys.stdout.write(colored('%s' % sys_string + c, 'yellow'))
		sys_string += c
		sys.stdout.flush()
		time.sleep(delay)

def delay_print_system(s, delay):
	# Similar to delay_print(), but implemented to extend its functionality. 
	growingMessage = ''
	for c in s:
		sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", growingMessage), 70, subsequent_indent="                      ")
		sys.stdout.write(colored('%s' % sys_string + c, 'yellow'))
		growingMessage += c
		sys.stdout.flush()
		time.sleep(delay)
	sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", growingMessage), 70, subsequent_indent="                      ")
	sys.stdout.write(colored('%s\n' % sys_string + c, 'yellow'))

def printError(msg):
	sys.stdout.write(colored("\r%20s" % username + ": ")) #display the user input line in the console.
	if slowResponse == True:
		time.sleep(RESPONSE_TIME)
	#sys.stdout.write(colored("\r" + textwrap.fill("{:>20}: {:<60}\n".format(BOTNAME, response), 70, subsequent_indent="                      ") + "\n", 'cyan')) # Replace the "user input line" with the bots' response

	sys_string = "\r" + textwrap.fill("{:>20}: {}".format("ERROR", msg), 70, subsequent_indent="                      ")
	sys.stdout.write(colored('%s\n' % sys_string, 'red'))
def printSystem(msg):
	sys.stdout.write(colored("\r%20s" % username + ": ")) #display the user input line in the console.
	if slowResponse == True:
		time.sleep(RESPONSE_TIME)
	# delay_print_system(msg, 0.1)
	sys_string = "\r" + textwrap.fill("{:>20}: {}".format("System", msg), 70, subsequent_indent="                      ")
	sys.stdout.write(colored('%s\n' % sys_string, 'yellow'))
	
	#sys.stdout.write(colored('%s\n' % sys_string, 'yellow'))
# END OF DOCUMENT
# Load all functions, then run the main function (removes need for forward declarations,
# which don't exist in Python)
if __name__=="__main__":
   main()