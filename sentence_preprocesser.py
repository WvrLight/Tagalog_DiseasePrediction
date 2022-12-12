from TglStemmer import stemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json

with open("{}/{}.json".format('cfg', "stopwords-tl")) as json_file:
    stopwords = json.load(json_file)

def stem_sentence(sentence):
    return stemmer('2', sentence, '1')

def remove_stopwords(tokenizedSentence):
    for stopword in stopwords:
        for word in tokenizedSentence:
            if (word == stopword):
                tokenizedSentence.remove(word)
    return tokenizedSentence

def preprocess_sentence(sentence):
    stemmed_sentence = stem_sentence(sentence)
    cleaned_sentence = remove_stopwords(stemmed_sentence)

    untokenized_sentence = ""
    for token in cleaned_sentence:
        untokenized_sentence += token + " "
    print(untokenized_sentence)


#inputSentence = input("Enter sentence with symptom descriptions: ")
inputSentence = "Ako ay inuubo at masakit ang ulo."
inputSentence = inputSentence.lower()

preprocess_sentence(inputSentence)
