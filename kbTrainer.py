#!python3
import os
import glob
import re
import nltk
import string
import pickle
import verb
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
# from brain import Brain, BrainEntry

####################################
#            HER BRAIN
####################################

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
		print(cosine_similarities)
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


#add global variable of token dictionary for tf-idf matrix reference


def main():
	# Get dictionary of KB dictionary files
	kbDict = getKBDict()
	for key, value in kbDict.items():
		print(key, value['title'])

	for key, value in kbDict.items():
		kbDict[key]['resolution'] = processResolution(kbDict[key]['resolution'])
		print(kbDict[key]['resolution'])

	kbWordsDict = getKBWordsDict(kbDict, 'issues')

	#STUB
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
	myBrain = Brain(kbWordsDict, {'KB00206580':myBrainEntry})

	brain_file = open("picklejar/brain.pickle", "wb")
	pickle.dump(myBrain, brain_file)
	brain_file.close()

def getKBWordsDict(kbDocuments, focus):
	kbWordsDict = OrderedDict()
	#For each KB Document take the dictionary from the KB
	for documentTitle, dictionary in kbDocuments.items():
		#For each string in its current focus i.e, (environment, issues, causes)
		#TODO: Must account for title, resolution cases 
		for string in dictionary[focus]:
			text = preprocessString(string)
			#Insert preprocessed text into token_tfidf dictionary
			kbWordsDict[documentTitle] = text
	return kbWordsDict

def preprocessString(string):
	#Preprocessing text by making them lower and remove punctuation
	text = string.lower()
	# TODO: Remove punctuation. Line below doesn't work in Python 
	text = re.sub(r'[\.,-\?]', '', text)
	return text

def getKBDict():
	''' Parses each KB text file in 'kb' directory into a
	dictionary with keys and values for each section of the article.
	Returns a dictionary of these kb dictionaries with
	key values = the KB id of each article.
	Ex: {'KB012345':{
				 'title':[...],
				 'environment':[...]
				 'issues':[...]
				 'causes':[...]
				 'resolution':[['...', 0], ['...', 0], ['...', 1]]'
			},
		 'KB022222':{
				 'title':[...],
				 'environment':[...]
				 'issues':[...]
				 'causes':[...]
				 'resolution':[['...', 0], ['...', 1], ['...', 1]]'
			}
		}
	'''
	###########################
	# KB Section Regex Objects
	###########################
	regIdTitle = re.compile(r'(KB.*): (.*)')
	regBlankLine = re.compile(r'^$')
	regEnvironment = re.compile(r'^(.*)')
	regIssue = re.compile(r'^- (.*)')
	regCause = re.compile(r'^- (.*)')
	regResolution = re.compile(r'^\s*(.). (.*)')

	# Get paths to all files in kb subdirectory
	kbArticlePaths = glob.glob('kb/*')

	# Dictionary to store KB articles
	kbDict = OrderedDict()

	# Load contents of each file
	for articlePath in kbArticlePaths:
		# Declare variables
		id = '' # Stores the key of the article (ex: 'KB0012345')
		title = '' # Stores the title of the article (ex: 'email access')
		environment = [] # Stores a list of the Environment tags (ex: ['Employee email system', 'Agent email system'])
		issues = []	# Stores a list of the Issue/Error entries (ex: ['User is not able to login to email system via Outlook', 'User is not able to login to web-email from a web browser'])
		causes = []	# Stores a list of the Cause entires (ex: ['User entered incorrect ID and Password.', 'User email account is locked after three consecutive failed login attempts.'])
		resolution = [] # Stores a LIST OF LISTS. Lists contain [text, level], where 'text' is the resolution text and 'level'
					    # is the list level of the entry (either 0 or 1).
					    # ex: [['Resolve the incident by unlocking the email account.', 0],
					    #	   ['Verify with the user that their issue has been resolved.', 0],
					    #	   ['Help the user understand and correct the root cause if necessary.', 0],
					    #	   ['If the user has a mobile device connected to their work email, ...', 1]]
					    # Note that a list entry with a level == 1 is a child step of the most recent step with a level == 0

		# track which section of the article is currently being read
		currentSection = 'initial'

		# Parse the text of each article into respective variables/lists
		with open(articlePath, encoding = 'utf-8') as f:
			# parse articleText
			for line in f:
				line = line.replace(u"’", u"'")
				line = line.encode('ascii', 'ignore').decode('ascii').rstrip('\n')
				if regBlankLine.match(line):
					continue
				else:
					if currentSection == 'initial':
						if line == 'Environment':
							currentSection = 'environment'
							continue
						else:
							m = regIdTitle.match(line)
							if m:
								id = m.group(1)
								title = m.group(2)
					elif currentSection == 'environment':
						if line == 'Issue/Error':
							currentSection = 'issue'
							continue
						else:
							m = regEnvironment.match(line)
							if m:
								environment.append(m.group(1))
					elif currentSection == 'issue':
						if line == 'Cause':
							currentSection = 'cause'
							continue
						else:
							m = regIssue.match(line)
							if m:
								issues.append(m.group(1))
					elif currentSection == 'cause':
						if line == 'Resolution/Workaround':
							currentSection = 'resolution'
							continue
						else:
							m = regCause.match(line)
							if m:
								causes.append(m.group(1))
					elif currentSection == 'resolution':
						m = regResolution.match(line)
						if m:
							level = (0 if m.group(1).isdigit() else 1)
							text = m.group(2)
							resolution.append([text, level])

		kbArticle = {'title':title, 'environment':environment, 'issues':issues, 'causes':causes, 'resolution':resolution}

		# Add KB article to dictionary
		kbDict[id] = kbArticle

	return kbDict

def processResolution(steps):
	modSteps = []
	for step, level in steps:
		words = word_tokenize(step)
		tagged_words = nltk.pos_tag(words)

		statement = []
		modStep = None

		# 'Convert 'resolve' step to first person
		if tagged_words[0][0] == "Resolve":
			verb_index = None
			statement = ['I', 'have', ]
			for i, tagged_word in enumerate(tagged_words):
				word = tagged_word[0]
				tag = tagged_word[1]
				if tag == 'VBG':
					verb_index = i
					break;
			if verb_index:
				verb_past = getPastParticiple(words[verb_index])
				statement.append(verb_past)
				statement.extend(words[verb_index+1:-1])
				statement.extend(['for', 'you'])
			modStep = ' '.join(statement) + '.'
		# Convert 'Verify' step into proper form
		elif tagged_words[0][0] == "Verify":
			statement = []
			#######################################################################################
			# Determine whether to ask Yes/No question or to ask for specific piece of information
			######################################################################################
			questionType = None
			if 'or' in words: # Asking about one thing OR another (get input). Have to consider if word 'also' or synonym is found
				questionType = 'OPTION'
			else:
				for word, tag in tagged_words[1:]:
					# Yes/No - Look for past participles or gerunds (is it 'working', issue has 'been' 'resolved')
					
					if tag in ('VB', 'VBD', 'VBG', 'VBN'):
						questionType = 'YN'
						break;
			if not questionType:
				questionType = 'DOMAIN'

			if questionType == 'YN':
				possessive_index = None
				nouns = []
				verb_index = None

				# Find start of useful info in step
				# TODO: Refactor this for loop into a function that returns the possessive index
				for i, tagged_word in enumerate(tagged_words):
					word = tagged_word[0]
					tag = tagged_word[1]
					if tag in ('POS', 'PRP$'): # possessive ending and possessive pronoun POS tags
						possessive_index = i
				# Find the subject (noun) and verbs
				for i, tagged_word in enumerate(tagged_words[possessive_index+1:]): # Scan through everything after the possessive noun, which is assumed to be "user's", "customer's", "their", or similar
					word = tagged_word[0]
					tag = tagged_word[1]
					if tag in ('NN', 'NNS'): # noun and plural noun
						nouns.append(word)
					if tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBZ') and not verb_index:
						verb_index = i


				if (possessive_index):
					statement.extend(['Please', 'verify', 'that', 'your'])
					statement.append(' and '.join(nouns))
					statement.extend(words[possessive_index+1+verb_index:-1])

				modStep = ' '.join(statement) + '.'

		if modStep:
			print(">>", modStep)
			modSteps.append(modStep)
		else: 
			modSteps.append(step)
	return modSteps




def getPastParticiple(vbg):
	vbn = verb.verb_past(vbg)
	return vbn


if __name__=="__main__":
   main()