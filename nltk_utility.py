import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    
    return nltk.word_tokenize(sentence)


def stem(word):

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):

    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence: 
            bag[idx] = 1

    return bag



# # TESTING

# a = "What services do you provide?"
# print(a)
# a = tokenize(a)
# print(a)

# b=["running", "runs", "runner", "ran", "dogs", "cats", "leaves", "swimming", "swam", "organization", "organize", "organizes"]
# print(b)
# stem_b = [stem(word) for word in b]
# print(stem_b)

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog = bag_of_words(sentence, words)
# print(bog)