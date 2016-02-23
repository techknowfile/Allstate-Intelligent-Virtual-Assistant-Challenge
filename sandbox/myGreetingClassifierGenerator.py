import nltk
import random
from nltk.corpus import nps_chat
from nltk.corpus import stopwords
import pickle

stop_words = set(stopwords.words('english'))
stop_words.remove('no')
stop_words.add('...')

xml_posts_0 = nps_chat.xml_posts()
posts_0 = nps_chat.posts()

categorized_posts = []
index = 0
# Categorize 'Greeting' and 'Non-greeting' posts
for el in xml_posts_0:
    if el.attrib.get('class') == 'Greet':
        categorized_posts.append((posts_0[index], 'Greeting'))
    else:
        categorized_posts.append((posts_0[index], 'Non-greeting'))
    index += 1

all_words = []
for (post, category) in categorized_posts:
    for word in post:
        all_words.append(word.lower())

all_words = nltk.FreqDist(w.lower() for w in all_words if len(w) > 1 if w not in stop_words)
word_features = [word[0] for word in all_words.most_common(50)]

def find_features(post):
    words = set(post)
    features = {}
    for w in word_features:
        if len(w) > 1:
            features[w] = (w in words)
    return features

random.shuffle(categorized_posts)

featuresets = [(find_features(post), category) for (post, category) in categorized_posts]

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

############################################################
# Show accuracy of the classifier
# ############################################################
print("Naive Bayes Algo accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(10)

class featureClassificationObject:
    def __init__(self, word_features, classifier):
        self.word_features = word_features
        self.classifier = classifier
    def find_features(post):
        words = set(post)
        features = {}
        for w in self.word_features:
            if len(w) > 1:
                features[w] = (w in words)
        return features

myYesNoAnswerClassificationObject = featureClassificationObject(word_features, classifier)


save_classifier = open("myGreetingClassificationObject.pickle", "wb")
pickle.dump(myYesNoAnswerClassificationObject, save_classifier)
save_classifier.close()