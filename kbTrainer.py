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
	kbDict = getKBDict()
	for key, value in kbDict.items():
		print(key, value['title'])

	for key, value in kbDict.items():
		kbDict[key]['resolution'] = processResolution(kbDict[key]['resolution'])
		print(kbDict[key]['resolution'])

	kbWordsDict = getKBWordsDict(kbDict, ['title', 'issues', 'causes'])

	#STUB
	myBrainEntry0 = BrainEntry(
		{'billing_address':[], 'mailing_address':[]},
		{'billing_address_bool':None, 'mailing_address_bool':None},
		[
			['What is your current billing address?', 'add_domain_knowledge', 'billing_address'],
			['What is your current mailing address?', 'add_domain_knowledge', 'mailing_address'],
			['Are you requesting to change your billing address only or also your mailing address?', 'or_bool', (('billing_address_bool', 'billing address'),('mailing_address_bool', 'mailing address'))],
			['What would you like your new billing address address to be?', 'conditional_update_domain_knowledge', 'billing_address', 'billing_address_bool'],
			['What would you like your new mailing address to be?', 'conditional_update_domain_knowledge', 'mailing_address', 'mailing_address_bool'],
			['', 'confirm']
		],
		None # stub for loop step
	)
	myBrainEntry1 = BrainEntry(
		{},
		{'issue_bool':None},
		[
			['I have unlocked your email account', 'resolve'],
			['Please verify that your issue has been resolved', 'yes_no', 'issue_bool'],
		],
		None # stub for loop step
	)
	myBrainEntry2 = BrainEntry(
		{},
		{'computer_bool':None},
		[
			['Please verify that your computer is plugged in.', 'yes_no', 'issue_bool'],
		],
		None # stub for loop step
	)
	brainEntryDict = {'KB00206580':myBrainEntry0, 'KB0083060':myBrainEntry1, 'KB00001337': myBrainEntry2}
	myBrain = Brain(kbWordsDict, brainEntryDict)

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
		new_bool_var = None


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
				for i, tagged_word in enumerate(tagged_words):
					word = tagged_word[0]
					tag = tagged_word[1]
					if tag == 'VBG':
						verb_index = i
						break;



				or_index = None 
				noun_before = ''
				noun_after = ''
				possessive_index = None
				nouns_after = []
				nouns_before = []


				# split into two strings: before and after 'or' 
				for i, tagged_word in enumerate(tagged_words):
					word = tagged_word[0]
					tag = tagged_word[1]
					if word == 'or':
						str_before_or = ' '.join(words[:i])
						str_after_or = ' '.join(words[i+1:]) # or only i if using split function 
						#str_after_or = str_after_or.split('or') 
						or_index = i
						break
				
				# split to put into list form 
				str_before_or = str_before_or.split()
				str_after_or = str_after_or.split()

				# get pos tagging 
				str_before_or = nltk.pos_tag(str_before_or)
				str_after_or = nltk.pos_tag(str_after_or)


				#######################################################
				# PROBS DONT NEED THIS ANYMORE BECAUSE OF CHUNKING LOL
				#######################################################

				# find all nouns in both strings 
				nouns_before = findNouns(str_before_or) # need to add matching since as of now, 'customer change' (first pairing) is returned
				nouns_after = findNouns(str_after_or)

				
				# find noun that matches stored domain knowledge 
				#noun_before = findDomainKnowledge(nouns_before, False)
				#noun_after = findDomainKnowledge(nouns_after, False)

				# get new variable created from this domain knowledge
				first_domain_var = findDomainKnowledge(nouns_before, True)
				second_domain_var = findDomainKnowledge(nouns_after, True)


				#######################################
				#  TODO: Get Nouns (using or to split sides)
				#######################################
				# use function to get noun before or and noun after or as two variables
				# these go into ((noun_1_bool, noun 1), (noun_2_bool, noun 2)) format
				# and get added to step_list[2] at the end

				##################################################
				# add noun_1_bool and noun_2_bool to dictionary
				##################################################
				knowledge_dict[noun_before] = None
				knowledge_dict[noun_after] = None

				#######################################
				#  TODO: Get possessive word
				#######################################
				possessive_index = getPossessiveIndex(enumerate(tagged_words))

				step = step.replace(words[possessive_index], 'your')
				words = word_tokenize(step)

				########################################
				# Build statement
				########################################
				statement.extend(['Are', 'you'])
				statement.append(words[verb_index])
				statement.extend(words[verb_index+1:])

				modStep = ' '.join(statement) + '?'

				#print(modStep)



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
				var_found = True
				possessive_index = None
				nouns = []
				verb_index = None
				stored_info = [] # input that needs to be verified/returned to the user as output
 
				possessive_index = getPossessiveIndex(enumerate(tagged_words))
 
				for i, tagged_word in enumerate(tagged_words[possessive_index+1:]): # scan after possessive noun
					word = tagged_word[0]
					tag = tagged_word[1]
 
					if tag in ('NN', 'NNS'): # noun and plural noun
						nouns.append(word)
					if tag in ('VB', 'VBD', 'VBG', 'VBN', 'VBZ') and not verb_index:
						verb_index = i
 
				# look through list of nouns to find variables that need to be created
				# assumption: steps involving domain knowledge will only contain one noun/variable at a time (ie verify the customer's x)
				for i, word in enumerate(nouns):
					if (nouns[i+1]):
						next_word = nouns[i+1]
						new_domain_var = word + '_' + next_word
						word_to_store =  word + ' ' + next_word
						knowledge_dict[new_domain_var] = None
						break
					elif not next_word: # in the off chance that the noun is not compound, only store the first noun
						new_domain_var = word
						word_to_store =  word
						knowledge_dict[new_domain_var] = None
 
				statement.extend(['Can', 'I', 'verify', 'your', ])
				statement.append(word_to_store)
				modStep = ' '.join(statement) + '?'
 
				step_type = 'add_domain_knowledge'
				new_domain_var = ''





			################################################
			#             Yes/No Question
			################################################
			elif questionType == 'YN':
				possessive_index = None
				nouns = []
				verb_index = None

				# Find start of useful info in step
				possessive_index = getPossessiveIndex(enumerate(tagged_words))
	
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

				else:
					statement.extend(words)

				modStep = ' '.join(statement) + '.'
				step_type = 'yes_no'

				print(modStep, 'test')


		################################################
		#            Confirm updates/changes
		################################################
		elif tagged_words[0][0] == 'Confirm':
			modStep = ''
			step_type = 'confirm'

		

		# Append items to step_list
		if modStep:
			step_list.append(modStep)
		if step_type:
			step_list.append(step_type)
		# TODO: Append rest of items
		# Append step_list to steps_list
		steps_list.append(step_list)

			
	# STUB	
	myBrainEntry = BrainEntry(knowledge_dict, bool_dict, steps_list, None)
 
	for key, value in knowledge_dict.items():
		print(key, value)



# find domain knowledge in a list of nouns - get_var is a bool value that determines whether or not the returned value 
# should be in the form of a noun with an underscore or space 
def findDomainKnowledge(noun_list, get_var): 
	for i, word in enumerate(noun_list):
		if (noun_list[i+1]):
			next_word = noun_list[i+1]
			new_domain_var = word + '_' + next_word
			word_to_store =  word + ' ' + next_word
			#knowledge_dict[new_domain_var] = None
			if get_var is False:
				return word_to_store
			elif get_var is True:
				return new_domain_var
			break
		elif not next_word: # in the off chance that the noun is not compound, only store the first noun
			new_domain_var = word
			word_to_store =  word
			#knowledge_dict[new_domain_var] = None

	return word_to_store


def extractChunk(t):
            entities = []
            if hasattr(t, 'label') and t.label() == 'Chunk':
                entities.append(' '.join(c[0] for c in t.leaves()))
            else:
                for child in t:
                    if not isinstance(child, str):
                        entities.extend(extractChunk(child))

            return entities


def orStepParser(sentence):
    words = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(words)

    chunkGram = r"""Chunk: {<POS|PRP\$><.*>*<NN.?>+<.*>*<CC><.*>*<NN.?>+}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)

    nounChunkGram = r"""Chunk: {<NN.?>+}"""
    nounChunkParser = nltk.RegexpParser(nounChunkGram)

    possessiveChunkGram = r"""Chunk: {<POS|PRP\$>}"""
    possessiveParser = nltk.RegexpParser(possessiveChunkGram)

    gerundActionGram = r"""Chunk: {<VBG><.*>*}(?:<POS|PRP\$><.*>*<CC>)"""
    gerundActionParser = nltk.RegexpParser(gerundActionGram)
    gerundActionChunked = gerundActionParser.parse(tagged)

    returnList = []

    main_entities = []
    main_entities.extend(extractChunk(chunked))

    gerund_action_entities = []
    gerund_action_entities.extend(extractChunk(gerundActionChunked))
    
    noun_entities = []

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
    chunked = nounChunkParser.parse(tagged_words)
    noun_entities.extend(extractChunk(chunked))

    nounKeyList = []
    for noun_entity in noun_entities:
        nounTuple = (noun_entity.replace(" ", "_"), noun_entity)
        nounKeyList.append(nounTuple)
    nounKeyTuple = tuple(nounKeyList)
        
    returnList.append(new_statement)
    returnList.append(nounKeyTuple)

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