#!python3
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

issueDetermined = False
issueResolved = False
kbArticleID = -1

# CONSTANTS
BOTNAME = "Alice (TSR)"
USERNAME = "User"

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
		self.resolution = resolution;

kbKeywordDict = {'address':(9, 'KB00206580'), 'login':(7, 'KB0083060')}

kb1 = KBArticle('''1. Resolve the incident by unlocking the email account.
2. Verify with the user that their issue has been resolved.
3. Help the user understand and correct the root cause if necessary.
   a. If the user has a mobile device connected to their work email, verify that the user has updated the password on their mobile device as well, since that may be the root cause of the account becoming locked.
''')

kb2 = KBArticle('''Resolution/Workaround
1. Verify the customer’s current billing address
2. Verify the customer’s current mailing address
3. Verify that the customer is requesting a change to their billing address only or also their mailing address
4. Update their billing address
5. If the customer would like to update their mailing address as well, update their mailing address
6. Confirm the customer’s new billing address and, if applicable, their mailing address after the change'''
)

kbLibrary = {'KB0083060':kb1, 'KB00206580':kb2}

##############################################
# Start of main code
##############################################
def main():
	''' TODO: Docstring'''
	clearScreen();

	while not issueResolved:
		kbArticle = determineIssue() #Gets Key
		print(kbArticle)

	tellUser("Is there anything else I can help you with today?")

def determineIssue():
	tellUser("How can I help you?")
	userInput = tellBot() # Get user input
	keywords = parseInput(userInput) # parse keywords from user input
	
	matched_keywords = [(keyword, kbKeywordDict.get(keyword)) for keyword in keywords if keyword in kbKeywordDict]
	matched_keywords = sorted(matched_keywords, key=lambda keyword: keyword[1][0], reverse=True)
	
	if matched_keywords:
		# keyword matched!
		issueDetermined = True
		# Returns only the highest rated match. Remove [0] to return all matched Keyword/KB article pairs
		return matched_keywords[0][1]

	if issueDetermined:
		pass
	
	else:
		tellUser("I'm sorry, I'm not able to assist with that issue.")


def tellUser(response):
	print(BOTNAME + ":", response)

def tellBot():
	return input(USERNAME + ": ")

def clearScreen():
	''' Function to clear current screen based on current OS '''
	os.system('cls' if os.name == 'nt' else 'clear')

def parseInput(userInput):
	# Tokenize user input into a list of words.
	words = word_tokenize(userInput)
	# Remove stop words and words less than 2 characters long, then LEMMATIZE input
	#words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words and len(w) > 1] 
	return words


# END OF DOCUMENT
# Load all functions, then run the main function (removes need for forward declarations,
# which don't exist in Python)
if __name__=="__main__":
   main()