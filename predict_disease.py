# IMPORTS

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Embedding, Bidirectional
from tf2crf import CRF, ModelWithCRFLoss
from TglStemmer import stemmer
import re
from nltk import word_tokenize

# CFG

import json

folder_name = 'cfg'

with open("{}/{}.json".format(folder_name, "word_list")) as json_file:
    wordIdList = json.load(json_file)

with open("{}/{}.json".format(folder_name, "ner_config")) as json_file:
    ner_config = json.load(json_file)

with open("{}/{}.json".format(folder_name, "tags")) as json_file:
    tags = json.load(json_file)

with open("{}/{}.json".format(folder_name, "symptom_list")) as json_file:
    symptom_list = json.load(json_file)

with open("{}/{}.json".format(folder_name, "disease_list")) as json_file:
    disease_list = json.load(json_file)

with open("{}/{}.json".format(folder_name, "stopwords-tl")) as json_file:
    stopwords = json.load(json_file)

PERCENTAGE_LOWER_LIMIT = 0.2

# MAIN FUNCTION

def predict_disease(ner_model, classification_model, raw_text, preprocessing_option):
    # Load the models
    #ner_model, classification_model = load_models()

    # Separate the raw text into multiple sentences if necessary
    raw_text = raw_text.lower()
    if (raw_text[len(raw_text) - 1] != '.'): raw_text = raw_text.join('.')
    sentence_list = re.split(r' *[\.\?!][\'"\)\]]* *', raw_text)
    del sentence_list[len(sentence_list) - 1]

    # Get the sentence inputs and pre-process them if selected
    tokenized_sentence_list = []
    id_sentence_list = []
    for sentence in sentence_list:
        if (preprocessing_option == 1):
            stemmed_sentence = stemmer('2', sentence, '1')
            tokenized_sentence = remove_stopwords(stemmed_sentence)
            tokenized_sentence_list.append(tokenized_sentence)
        else:
            tokenized_sentence = word_tokenize(sentence)
            tokenized_sentence_list.append(tokenized_sentence)

        id_per_word = convert_sentence_to_idx(tokenized_sentence)
        id_sentence_list.append(id_per_word)

    # Get the symptoms of each sentence
    symptom_group = []
    i = 0
    for sentence in id_sentence_list:
        recognized_symptoms = recognize_symptoms_in_sentence(ner_model, sentence, tokenized_sentence_list[i])
        symptom_group.append(symptoms_to_boolean(recognized_symptoms))
        i += 1
    merged_symptom_info = merge_symptom_info(symptom_group)

    # Using the symptom info, predict possible diseases
    prediction_result = analyze_symptoms(classification_model, merged_symptom_info)
    rounded_results = []
    for probability in prediction_result:
        rounded_results.append(np.round(probability, decimals=6))

    # TODO: return list of diseases and probabilities for UI
    # Create a list of diseases and the given probability
    symptom_results = []
    for i in range(len(symptom_list)):
        if (merged_symptom_info[i] == 1):
            symptom_results.append(symptom_list[i])

    disease_results = {}
    for key, val in disease_list.items():
        if (val < len(rounded_results[0]) and rounded_results[0][val] > PERCENTAGE_LOWER_LIMIT):
            disease_results.update({key: rounded_results[0][val]})

    print(rounded_results[0])
    return symptom_results, disease_results

# FUNCTIONS


def load_models(model_type):
    import pickle

    # ner_model = tf.keras.models.load_model(model_path)
    ner_model = initialize_ner(model_type)
    ner_model.summary()

    with open('naiveBayes.pkl', 'rb') as f:
        naiveBayes = pickle.load(f)

    return ner_model, naiveBayes

def initialize_ner(model_type):
    inputs = tf.keras.layers.Input(shape=(None,), dtype='int32')
    output = Embedding(ner_config['n_words'], ner_config['word_embedding_size'],
                       trainable=True, mask_zero=True)(inputs)
    if (model_type == 'lstm'):
        bi_rnn = Bidirectional(LSTM(units=ner_config['word_embedding_size'],
                                    return_sequences=True,
                                    dropout=0.5,
                                    recurrent_dropout=0.5,
                                    kernel_initializer=tf.keras.initializers.he_normal()))(output)
        rnn = LSTM(units=ner_config['word_embedding_size'] * 2,
                   return_sequences=True,
                   dropout=0.5,
                   recurrent_dropout=0.5,
                   kernel_initializer=tf.keras.initializers.he_normal())(bi_rnn)
    else:
        bi_rnn = Bidirectional(GRU(units=ner_config['word_embedding_size'],
                                   return_sequences=True,
                                   dropout=0.5,
                                   recurrent_dropout=0.5,
                                   kernel_initializer=tf.keras.initializers.he_normal()))(output)
        rnn = GRU(units=ner_config['word_embedding_size'] * 2,
                  return_sequences=True,
                  dropout=0.5,
                  recurrent_dropout=0.5,
                  kernel_initializer=tf.keras.initializers.he_normal())(bi_rnn)
    crf = CRF(units=ner_config['n_tags'], dtype='float32')
    output = crf(rnn)
    base_model = Model(inputs, output)
    ner_model = ModelWithCRFLoss(base_model, sparse_target=True)
    ner_model.build(ner_config['shape'])

    if (model_type == 'lstm'):
        ner_model.load_weights("bilstm")
    else:
        ner_model.load_weights("bigru")
    return ner_model
    
def remove_stopwords(tokenizedSentence):
    for stopword in stopwords:
        for word in tokenizedSentence:
            if (word == stopword):
                tokenizedSentence.remove(word)
    return tokenizedSentence

def convert_sentence_to_idx(tokenizedSentence):
    END_IDX = ner_config["n_words"] - 2
    UNK_IDX = ner_config["n_words"] - 1

    sentence2idx = []
    for word in tokenizedSentence:
        wordFound = False
        for key, val in wordIdList.items():
            if (word == key):
                wordFound = True
                sentence2idx.append(val)
        if (not wordFound):
            sentence2idx.append(UNK_IDX)
    while (len((sentence2idx)) < ner_config["maxlen"]):
        sentence2idx.append(END_IDX)
    return sentence2idx

# For a given sentence, predict the words which are related to symptom information
def recognize_symptoms_in_sentence(ner_model, sentence2idx, tokenizedSentence):
    I_TAG_INDEX = 0

    p = ner_model.predict(np.array([sentence2idx]))
    #p = np.argmax(p, axis=-1)

    input_symptom_list = []

    # Iterate through the entire sentence
    for idx, (w, pred) in enumerate(zip(sentence2idx, p[0])):
        if (tags[pred] == 'B-SYMPTOM'):
            symptom_word = tokenizedSentence[idx]

            # Check for additional words for a symptom
            temp_idx = idx + 1
            if (temp_idx < (len(tokenizedSentence) - 1) and p[0][temp_idx] == I_TAG_INDEX):
                while (p[0][temp_idx] == I_TAG_INDEX):
                    symptom_word = symptom_word + " " + tokenizedSentence[temp_idx]
                    if (temp_idx != len(tokenizedSentence) - 1):
                        temp_idx += 1
                    else:
                        break
            input_symptom_list.append(symptom_word)
        if (idx == len(tokenizedSentence) - 1):
            break
    return input_symptom_list

# Match each symptom with the trained symptom list
def symptoms_to_boolean(input_symptom_list):
    from difflib import SequenceMatcher

    STRING_MATCH_PERCENTAGE = 0.75
    input_to_boolean = []

    for symptom in symptom_list:
        symptomInList = False
        for input_symptom in input_symptom_list:
            if (symptom in input_symptom or SequenceMatcher(None, input_symptom, symptom).ratio() >= STRING_MATCH_PERCENTAGE):
                symptomInList = True
        if (symptomInList):
            input_to_boolean.append(1)
        else:
            input_to_boolean.append(0)
    return input_to_boolean

# For each group, merge all existing symptoms into one array
def merge_symptom_info(symptom_arrays):
    # Initialize null array
    merged_symptoms = []
    for symptom in symptom_list:
        merged_symptoms.append(0)

    for symptom_group in symptom_arrays:
        for i, symptom in enumerate(symptom_group):
            if symptom == 1:
                merged_symptoms[i] = 1
    return merged_symptoms

# Run recognized symptoms through a classification model
def analyze_symptoms(classification_model, boolean_array_input):
    import pandas as pd
    from sklearn.preprocessing import RobustScaler

    #symptom_columns = []
    #for symptom in symptom_list:
        #symptom_columns.append([symptom, ],)

    #cols = pd.MultiIndex.from_arrays(symptom_columns)

    # input_frame = pd.DataFrame(boolean_array_input)
    
    input_array = np.array(boolean_array_input)
    input_array = input_array.reshape(1, -1)

    #scaler = RobustScaler()
    #scaled_frame = scaler.fit_transform(input_array)

    prediction_result = classification_model.predict_proba(input_array)
    return prediction_result