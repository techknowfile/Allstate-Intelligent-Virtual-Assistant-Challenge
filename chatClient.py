#!python3
import os
import nltk

issueDetermined = False
issueResolved = False
kbArticle = -1

# CONSTANTS
BOTNAME = "Alice (TSR)"
USERNAME = "User"

def main():
	''' TODO: Docstring'''
	clearScreen();

	while not issueResolved:
		kbArtcile = determineIssue()

def determineIssue():
	tellUser("How can I help you?")
	userInput = tellBot()
	print(userInput)

	if issueDetermined:
		pass
	
	else:
		tellUser("I'm sorry, I'm not able to assist with that issue.")
		tellUser("Is there anything else I can help you with today?")

def tellUser(response):
	print(BOTNAME + ":", response)

def tellBot():
	return input(USERNAME + ": ")

def clearScreen():
	''' Function to clear current screen based on current OS '''
	os.system('cls' if os.name == 'nt' else 'clear')

# END OF DOCUMENT
# Load all functions, then run the main function (removes need for forward declarations,
# which don't exist in Python)
if __name__=="__main__":
   main()