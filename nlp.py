import random
import json
import nltk
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("intents.json", encoding="utf-8") as f:

    data = json.load(f)

# Prepare dataset
patterns, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])

# NLP preprocessing + vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels

# Train classifier
clf = LogisticRegression()
clf.fit(X, y)

def best_match(user_inp, patterns):
    match = difflib.get_close_matches(user_inp, patterns, n=1, cutoff=0.6)
    return match[0] if match else None

import random

# Simple chatbot with memory
intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hey"],
        "responses": ["Hello!", "Hi there!", "Hey, how can I help you?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you", "good night"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    }
}

name = None  # memory variable

print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_inp = input("You: ").lower()
    if user_inp == "quit":
        break

    # 1. Capture name
    if "my name is" in user_inp or "i am" in user_inp:
        name = user_inp.split("is")[-1].strip().capitalize()
        print(f"Bot: Nice to meet you, {name}!")

    # 2. Recall name
    elif "what's my name" in user_inp or "what is my name" in user_inp or "whats my name" in user_inp or "do you know my name" in user_inp or "what my name" in user_inp:
        if name:
            print(f"Bot: Your name is {name}.")
        else:
            print("Bot: I don't know your name yet. Tell me by saying 'my name is ...'")

    # 3. Normal intents
    else:
        response = "Sorry, I didn't understand that."
        for intent, data in intents.items():
            for pattern in data["patterns"]:
                if pattern in user_inp:
                    response = random.choice(data["responses"])
                    break
        print("Bot:", response)


