#!python3
import os
import glob
import re

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

# Load contents of each file
for articlePath in kbArticlePaths:

	# Declare variables
	id = '' # Stores the key of the article (ex: 'KB0012345')
	title = '' # Stores the title of the article (ex: 'email access')
	environment = [] # Stores a list of the Environment tags (ex: ['Employee email system', 'Agent email system'])
	issue = []	# Stores a list of the Issue/Error entries (ex: ['User is not able to login to email system via Outlook', 'User is not able to login to web-email from a web browser'])
	cause = []	# Stores a list of the Cause entires (ex: ['User entered incorrect ID and Password.', 'User email account is locked after three consecutive failed login attempts.'])
	resolution = [] # Stores a LIST OF LISTS. Lists contain [text, level], where 'text' is the resolution text and 'level'
				    # is the list level of the entry (either 0 or 1).
				    # ex: [['Resolve the incident by unlocking the email account.', 0],
				    #	   ['Verify with the user that their issue has been resolved.', 0],
				    #	   ['Help the user understand and correct the root cause if necessary.', 0],
				    #	   ['If the user has a mobile device connected to their work email, ...', 1]]
				    # Note that a list entry with a level == 1 is a child step of the most recent step with a level == 0

	# track which section of the article is currently being read
	currentSection = 'initial'
	with open(articlePath, encoding = 'utf-8') as f:
		# parse articleText
		for line in f:
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
							print(id, title)
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
							issue.append(m.group(1))
				elif currentSection == 'cause':
					if line == 'Resolution/Workaround':
						currentSection = 'resolution'
						continue
					else:
						m = regCause.match(line)
						if m:
							cause.append(m.group(1))
				elif currentSection == 'resolution':
					m = regResolution.match(line)
					if m:
						level = (0 if m.group(1).isdigit() else 1)
						text = m.group(2)
						resolution.append([text, level])



