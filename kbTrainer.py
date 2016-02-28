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
				if boolKnowledgeDict.get('or_also_bool') == True:
					tellUser(step[0]) # Ask user for domain knowledge
					brainEntry.domainKnowledgeDict[step[2]].append(tellBot()) # Append domain knowledge to list for that specific domain knowledge (retains values)


class BrainEntry:
	def __init__(self, domainKnowledgeDict, boolKnowledgeDict, steps, loop_step):
		self.domainKnowledgeDict = domainKnowledgeDict
		self.boolKnowledgeDict = boolKnowledgeDict
		self.steps = steps


#add global variable of token dictionary for tf-idf matrix reference


def main():
	# Get dictionary of KB dictionary files
	# This just contains the text from each section broken up in
	# an accessible manner
	# DOES NOT GET ADDED TO BRAIN!
	kbDict = getKBDict()

	##########################################
	# Generate the KB Dictionary of knowledge
	# needed for the brain to help with issues
	##########################################
	brainEntryDict = {}
	for key, value in kbDict.items():
		aBrainEntry = processResolution(kbDict[key]['resolution'])
		brainEntryDict[key] = aBrainEntry

	#########################################
	# Generate a dictionary storing the words
	# in each KB section, keyed by the KB Key
	#
	# Needed by the Brain to match the user's
	# input to the correct KB article
	#########################################
	kbWordsDict = getKBWordsDict(kbDict, ['title', 'issues', 'causes'])

	#########################################
	# Build the Brain
	#########################################
	myBrain = Brain(kbWordsDict, brainEntryDict)

	#########################################
	# Pickle the Brain to be loaded into 
	# a chat client
	#########################################
	brain_file = open("picklejar/brain.pickle", "wb")
	pickle.dump(myBrain, brain_file)
	brain_file.close()

def getKBWordsDict(kbDocuments, focusList):
	kbWordsDict = OrderedDict()
	#For each KB Document take the dictionary from the KB
	for documentTitle, dictionary in kbDocuments.items():
		#For each string in its current focus i.e, (environment, issues, causes)
		#TODO: Must account for title, resolution cases 
		for focus in focusList:
			for string in dictionary[focus]:
				text = preprocessString(string)
				#Insert preprocessed text into token_tfidf dictionary
				if documentTitle in kbWordsDict:
					kbWordsDict[documentTitle] += ' ' + text
				else: 
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
	knowledge_dict = {}
	bool_dict = {}
	steps_list = []
	modSteps = []
	for step, level in steps:
		step_list = []

		statement = []
		step_type = ''
		new_domain_var = None
		update_domain_var = None
		new_bool_var = None # what we add to the bool dict
		ref_bool_var = None # what we look up in the bool dict
		if_question = None
		key_noun_pair_tuple = None

		words = word_tokenize(step)
		tagged_words = nltk.pos_tag(words)

		modStep = None

		# 'Convert 'resolve' step to first person
		if tagged_words[0][0] == 'Resolve':
			step_type = 'resolve'
			verb_index = None
			statement = ['I', 'have']
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
		elif tagged_words[0][0] == 'Verify':
			statement = []
			#######################################################################################
			# Determine whether to ask Yes/No question or to ask for specific piece of information
			######################################################################################
			questionType = None
			if 'or' in words: # Asking about one thing OR another (get input). Have to consider if word 'also' or synonym is found
				questionType = 'OPTION'

				if 'also' in words:
					step_type = 'or_also_bool'

				else: 
					step_type = 'or_bool'
			
				# get verb index
				# for i, tagged_word in enumerate(tagged_words):
				# 	word = tagged_word[0]
				# 	tag = tagged_word[1]
				# 	if tag == 'VBG':
				# 		verb_index = i
				# 		break;

				orStepList = orStepParser(step)
				for key_tuple in orStepList[1]:
					bool_dict[key_tuple[0]] = None
				# Set variables for step_list
				modStep = orStepList[0]
				key_noun_pair_tuple = orStepList[1]


				



			################################################
			#        Set Question type:Yes/No
			################################################
			else:
				for word, tag in tagged_words[1:]:
					# Yes/No - Look for past participles or gerunds (is it 'working', issue has 'been' 'resolved')
					
					if tag in ('VB', 'VBD', 'VBG', 'VBN'):
						questionType = 'YN'
						break

			################################################
			#           Add Domain Knowledge
			################################################	
			if not questionType:
				questionType = 'DOMAIN'

			if questionType == 'DOMAIN':
				returnList = addDomainStepParser(step)
				
				# Set step_list variables
				step_type = 'add_domain_knowledge'
				modStep = returnList[0]
				new_domain_var = returnList[1]
				# Add new domain var to dictionary
				knowledge_dict[new_domain_var] = None

			################################################
			#             Yes/No Question
			################################################
			elif questionType == 'YN':

				returnList = yesNoStepParser(step)

				# Set step_list variables
				step_type = 'yes_no'
				modStep = returnList[0]
				new_bool_var = returnList[1]

				# Add new bool to dictionary
				bool_dict[new_bool_var] = None



		################################################
		#    Update Domain Knowledge
		################################################
		elif tagged_words[0][0] == 'Update':
			updateStepList = updateStepParser(step)

			# Set variables for step_list
			modStep = updateStepList[0]
			step_type = 'update_domain_knowledge'
			update_domain_var = updateStepList[1]

		################################################
		#    Conditional Update Domain Knowledge
		################################################
		elif tagged_words[0][0] == 'If':
			# Get if statement
			if_statements = step.split(',')
			if_statement = nltk.word_tokenize(if_statements[0])
			tagged_words = nltk.pos_tag(if_statement)

			# Get nouns in if statement
			nouns = getNounNamedEntities(tagged_words)
			keyed_nouns = [noun.replace(" ", "_") for noun in nouns]
			# get keys for possible noun_bools
			keyed_noun_bools = [noun + "_bool" for noun in keyed_nouns]

			# Determine if noun_bool exists in dictionary.
			# If it does, it's a conditional_update_domain_knowledge step
			# If it does not, then it is an if_yes_no, meaning that we need
			# To ask a question, potentially create a new noun_bool,
			# check the statement that follows the comma, and look at its
			# first word to determine what to do next

			keyed_noun_bool_exists = False
			for noun_bool in keyed_noun_bools:
				if noun_bool in bool_dict:
					keyed_noun_bool_exists = True
					break

			###########################################
			# Complicated IF (noun_bool doesn't exist)
			###########################################
			if not keyed_noun_bool_exists:

				#######################################
				# Create if question
				#######################################
				ifChunkGram = r"""Chunk: (?:<VBZ>){<.*>*}"""
				ifChunkParser = nltk.RegexpParser(ifChunkGram)
				chunked = ifChunkParser.parse(tagged_words)

				statement_fragments = extractChunk(chunked)
				statement_fragment = statement_fragments[0].replace('their', 'your')

				modStep = "Do you have {}?".format(statement_fragment)

				# Set dictionary vars
				new_bool_var = keyed_noun_bools[-1]
				new_domain_var = keyed_nouns[-1]

				# Add to dictionaries
				bool_dict[new_bool_var] = None
				knowledge_dict[new_domain_var] = None

				####################################
				# Determin type of Complicated If
				####################################
				statement = if_statements[1].strip()
				words = nltk.word_tokenize(statement)
				firstWord = words[0]
				if firstWord.lower() == 'verify':
					returnList = yesNoStepParser(statement)
					
					# Set step_list variables
					step_type = 'if_yes_no_yes_no'
					if_question = returnList[0]
					new_bool_var = returnList[1]

					# Add new bool to dictionary
					bool_dict[new_bool_var] = None

				# if firstWord.

			elif keyed_noun_bool_exists:
				#####################################
				# noun_bool exists
				#####################################
				updateStepList = updateStepParser(step)

				# Set variables for step_list
				modStep = updateStepList[0]
				step_type = 'conditional_update_domain_knowledge'
				update_domain_var = updateStepList[1]
				ref_bool_var = updateStepList[1] + "_bool"


		################################################
		#            Confirm updates/changes
		################################################
		elif tagged_words[0][0] == 'Confirm':
			modStep = ' '
			step_type = 'confirm'

		

		# Append items to step_list
		if modStep:
			step_list.append(modStep)
		if step_type:
			step_list.append(step_type)
		if if_question:
			step_list.append(if_question)
		if new_domain_var:
			step_list.append(new_domain_var)
		if update_domain_var:
			step_list.append(update_domain_var)
		if new_bool_var:
			step_list.append(new_bool_var)
		if ref_bool_var:
			step_list.append(ref_bool_var)
		if key_noun_pair_tuple:
			step_list.append(key_noun_pair_tuple)

		print(step_list)

		# TODO: Append rest of items
		# Append step_list to steps_list

		steps_list.append(step_list)
	print("KD:", knowledge_dict)
	print("BD:", bool_dict)

	thisBrainEntry = BrainEntry(knowledge_dict, bool_dict, steps_list, None)
	return thisBrainEntry



def extractChunk(t):		
	entities = []
	# Check label of chunk
	if hasattr(t, 'label') and (t.label() == 'Chunk'):
		# Add chunk to entities list
		entities.append(' '.join(c[0] for c in t.leaves()))
	else:
		for child in t:
			if not isinstance(child, str):
				# Parse sub tree for new chunk
				entities.extend(extractChunk(child))

	return entities


def ifStatementParser(sentence):
	# Tokenize and tag

	words = nltk.word_tokenize(sentence)
	tagged_words = nltk.pos_tag(words)

	noun_entities = getNounNamedEntities(tagged_words)
	


def updateStepParser(sentence):
	# Tokenize and tag

	words = nltk.word_tokenize(sentence)
	tagged_words = nltk.pos_tag(words)

	noun_entities = getNounNamedEntities(tagged_words)
	noun = noun_entities[-1]
	noun_key = noun.replace(" ", "_")
	verb_entities = getVerbs(tagged_words)


	verb = getPastParticiple(verb_entities[0].lower())

	new_statement = 'What would you like your new {} to be?'.format(noun)
	returnList = [new_statement, noun_key]
	return returnList

def getNounNamedEntities(tagged_words):
	# define noun chunk and build nounChunkParser for named entity (noun) extraction
	nounChunkGram = r"""Chunk: {<NN.?>+}"""
	nounChunkParser = nltk.RegexpParser(nounChunkGram)
	chunked_nouns = nounChunkParser.parse(tagged_words)

	noun_entities = []

	noun_entities.extend(extractChunk(chunked_nouns))

	return noun_entities

def getVerbs(tagged_words):
	# define noun chunk and build nounChunkParser for named entity (noun) extraction
	verbChunkGram = r"""Chunk: {<VB.?>+}"""
	verbChunkParser = nltk.RegexpParser(verbChunkGram)
	chunked_verbs = verbChunkParser.parse(tagged_words)

	verb_entities = []

	verb_entities.extend(extractChunk(chunked_verbs))

	return verb_entities



def orStepParser(sentence):
	# Tokenize and tag
	words = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(words)

	# Get everything from the first possessive on into a chunk
	chunkGram = r"""Chunk: {<POS|PRP\$><.*>*<NN.?>+<.*>*<CC><.*>*<NN.?>+}"""
	chunkParser = nltk.RegexpParser(chunkGram)
	chunked = chunkParser.parse(tagged)

	# define possessive chunk and build possessiveParser for possessive pronoun/possessive ending extraction
	possessiveChunkGram = r"""Chunk: {<POS|PRP\$>}"""
	possessiveParser = nltk.RegexpParser(possessiveChunkGram)

	# Get everything from the gerund up to and excluding the first possessive (must be followed by conjunction [presumed 'or']
	gerundActionGram = r"""Chunk: {<VBG><.*>*}(?:<POS|PRP\$><.*>*<CC>)"""
	gerundActionParser = nltk.RegexpParser(gerundActionGram)
	gerundActionChunked = gerundActionParser.parse(tagged)

	# List of items to return
	returnList = []

	main_entities = []
	main_entities.extend(extractChunk(chunked))

	gerund_action_entities = []
	gerund_action_entities.extend(extractChunk(gerundActionChunked))

	main_entity = main_entities[0]

	words = nltk.word_tokenize(main_entity)
	tagged_words = nltk.pos_tag(words)
	possessive_entities = set()
	for tagged_word in tagged_words:
		if tagged_word[1] in ('POS', 'PRP$'):
			possessive_entities.add(tagged_word[0])

	for possessive in possessive_entities:
		main_entity = main_entity.replace(possessive, 'your')
	new_statement = 'Are you {} {}'.format(gerund_action_entities[0], main_entity)


	noun_entities = getNounNamedEntities(tagged_words)

	nounKeyList = []
	for noun_entity in noun_entities:
		nounTuple = (noun_entity.replace(" ", "_") + "_bool", noun_entity)
		nounKeyList.append(nounTuple)
	nounKeyTuple = tuple(nounKeyList)
		
	returnList.append(new_statement)
	returnList.append(nounKeyTuple)

	return returnList

def addDomainStepParser(sentence):
	# Tokenize and tag
	words = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(words)

	# Get everything after the first possessive on into a chunk
	chunkGram = r"""Chunk: (?:<POS|PRP\$>){<[^\.]*>]*}"""
	chunkParser = nltk.RegexpParser(chunkGram)
	chunked = chunkParser.parse(tagged)

	statements = extractChunk(chunked)
	statement = statements[0] # assuming there's only one statements (TODO: lol, this isn't getting fixed)

	words = nltk.word_tokenize(statement)
	tagged_words = nltk.pos_tag(words)

	# Get nouns
	noun_entities = getNounNamedEntities(tagged_words)
	noun_entity = noun_entities[0] # assuming there's only one named entity (TODO: lol, this isn't getting fixed)

	noun_key = noun_entity.replace(" ", "_")

	new_statement = "What is your {}.".format(statement)
	
	returnList = [new_statement, noun_key]
	return returnList

def yesNoStepParser(sentence):
	# Tokenize and tag
	words = nltk.word_tokenize(sentence)
	tagged = nltk.pos_tag(words)

	# Get everything after the first possessive on into a chunk
	chunkGram = r"""Chunk: (?:<POS|PRP\$>){<[^\.]*>]*}"""
	chunkParser = nltk.RegexpParser(chunkGram)
	chunked = chunkParser.parse(tagged)

	statements = extractChunk(chunked)
	statement_possessive_first = statements[0] # assuming there's only one statements (TODO: lol, this isn't getting fixed)

	# Get everything from the first first gerund on into a chunk
	chunkGram = r"""Chunk: {<VBN><[^\.]*>]*}"""
	chunkParser = nltk.RegexpParser(chunkGram)
	chunked = chunkParser.parse(tagged)

	# Get everything after the first possessive on into a chunk
	statements = extractChunk(chunked)
	statement_verb_first = statements[0] # assuming there's only one statements (TODO: lol, this isn't getting fixed)
	
	if len(statement_possessive_first) > len (statement_verb_first):

		statement = statement_possessive_first # assuming there's only one statements (TODO: lol, this isn't getting fixed)

		new_statement = "Please verify that your {}.".format(statement)
	
	else:
		statement = statement_verb_first
		new_statement = "Have you {}?".format(statement)

	# Get nouns

	words = nltk.word_tokenize(statement)
	tagged_words = nltk.pos_tag(words)
	noun_entities = getNounNamedEntities(tagged_words)
	noun_entity = noun_entities[0] # assuming there's only one named entity (TODO: lol, this isn't getting fixed)

	noun_bool = noun_entity.replace(" ", "_") + "_bool"

	returnList = [new_statement, noun_bool]
	return returnList


def findNouns(word_list):
	nouns = []
	for i, tagged_word in enumerate(word_list):	
		word = tagged_word[0]
		tag = tagged_word[1]
		if tag in ('NN', 'NNS'):
			nouns.append(word)

	return nouns



def getPossessiveIndex(words):
	for i, tagged_word in words:
		word = tagged_word[0]
		tag = tagged_word[1]
		if tag in ('POS', 'PRP$'): # possessive ending and possessive pronoun POS tags
			possessive_index = i
   
	return possessive_index


def getPastParticiple(vbg):
	vbn = verb.verb_past(vbg)
	return vbn


if __name__=="__main__":
   main()