import random
import json

import torch

from modelling import NeuralNet
from nltk_utility import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = data["model"] 
model.load_state_dict(model_state)
model.eval()

bot_name = "WasteBot"
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry, I didn't get that. I'm WasteBot, here to provide helpful information about e-waste recycling. You can ask me questions like: What is e-waste?, Why should I recycle e-waste?, What services do you offer?, help me indentify what can be considered as e-waste?"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

