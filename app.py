from flask import Flask, render_template, request, jsonify
import json, random, datetime, requests, wikipedia, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import difflib

# Load intents
with open("intents.json", encoding="utf-8") as f:
    data = json.load(f)

# Prepare dataset
patterns, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        labels.append(intent["tag"])

# Train ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = labels
clf = LogisticRegression()
clf.fit(X, y)

from textblob import TextBlob

import collections
# Memory and context
name = None
last_intent = None
last_entity = None  # like city name
conversation_history = []  # Stores tuples of (user, bot)
dialogue_state = None  # For managing multi-turn conversations
# Short-term memory for jokes, motivation, and greetings
joke_memory = collections.deque(maxlen=3)
motivation_memory = collections.deque(maxlen=3)
greeting_memory = collections.deque(maxlen=3)

# API keys
WEATHER_API_KEY = "19dbf69a51c79006168eea32b28674c8"

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    global name, last_intent, last_entity, conversation_history, dialogue_state
    user_inp = request.json["message"].lower().replace("'", "")
    # Fuzzy match correction for user input
    # Find closest pattern if input is a single word and not in patterns
    words = user_inp.split()
    corrected_words = []
    for w in words:
        # Only correct if not an exact match
        if w not in patterns:
            close = difflib.get_close_matches(w, patterns, n=1, cutoff=0.8)
            if close:
                # Use the closest match's word if it's a single word pattern
                # Otherwise, use original
                pattern_words = close[0].split()
                if len(pattern_words) == 1:
                    corrected_words.append(pattern_words[0])
                else:
                    corrected_words.append(w)
            else:
                corrected_words.append(w)
        else:
            corrected_words.append(w)
    corrected_inp = " ".join(corrected_words)
    if corrected_inp != user_inp:
        user_inp = corrected_inp
    # --- DuckDuckGo follow-up state ---
    if dialogue_state == "awaiting_more_info" and ("yes" in user_inp or "more" in user_inp or "tell me more" in user_inp):
        # Try to show more info from last DuckDuckGo query
        if hasattr(chatbot_response, "last_ddg_related") and chatbot_response.last_ddg_related:
            reply = chatbot_response.last_ddg_related
            dialogue_state = None
            return jsonify({"reply": reply})
        else:
            dialogue_state = None
            return jsonify({"reply": "Sorry, I don't have more details right now."})
    # Specific smalltalk Q&A
    if re.search(r"what can you do", user_inp):
        response = "I can chat, tell jokes, answer simple questions, and help you study!"
        conversation_history.append((user_inp, response))
        return jsonify({"reply": response})
    if re.search(r"who made you|who created you", user_inp):
        response = "I was made by a programmer for a lab project! 👩‍💻"
        conversation_history.append((user_inp, response))
        return jsonify({"reply": response})
    # Smalltalk and bot info
    if re.search(r"what can you do|who made you|who created you|tell me about yourself|do you like me|do you have feelings|are you real|are you human", user_inp):
        last_intent = "smalltalk"
        for intent in data["intents"]:
            if intent["tag"] == "smalltalk":
                response = random.choice(intent["responses"])
                conversation_history.append((user_inp, response))
                return jsonify({"reply": response})
    # --- Multi-turn dialogue and deeper context handling ---
    # Example: If user asks for a study tip, follow up with a related question
    if dialogue_state == "awaiting_study_topic":
        # User previously asked for study help, now expects a topic
        topic = user_inp.strip()
        response = f"Great! For {topic}, try breaking down the material into smaller sections and reviewing regularly. Do you want more tips for {topic}?"
        conversation_history.append((user_inp, response))
        dialogue_state = None
        return jsonify({"reply": response})

    # If user asks for study help, set state to await topic
    if re.search(r"study tip|study tips|help me study|exam|focus|prepare|revision", user_inp):
        last_intent = "study_help"
        dialogue_state = "awaiting_study_topic"
        for intent in data["intents"]:
            if intent["tag"] == "study_help":
                response = random.choice(intent["responses"])
                response += " What subject or topic do you need help with?"
                conversation_history.append((user_inp, response))
                return jsonify({"reply": response})
    # Example: If user asks about Python, offer to show code or answer follow-up
    if dialogue_state == "awaiting_python_example":
        if "yes" in user_inp or "show" in user_inp or "example" in user_inp:
            response = "Here's a simple Python example:\nprint('Hello, world!')"
            conversation_history.append((user_inp, response))
            dialogue_state = None
            return jsonify({"reply": response})
        else:
            response = "No problem! Let me know if you want to see a Python code example later."
            conversation_history.append((user_inp, response))
            dialogue_state = None
            return jsonify({"reply": response})

    if re.search(r"python", user_inp):
        if re.search(r"program|language|code|programming|script|software|developer", user_inp):
            response = "Python is a popular programming language known for its simplicity and versatility. Would you like a coding example?"
            conversation_history.append((user_inp, response))
            dialogue_state = "awaiting_python_example"
            return jsonify({"reply": response})
    # --- Keyword-based overrides for cognitive queries ---
    # Study help
    if re.search(r"study tip|study tips|help me study|exam|focus|prepare|revision", user_inp):
        last_intent = "study_help"
        for intent in data["intents"]:
            if intent["tag"] == "study_help":
                response = random.choice(intent["responses"])
                conversation_history.append((user_inp, response))
                return jsonify({"reply": response})

    # Python programming
    if re.search(r"python", user_inp):
        if re.search(r"program|language|code|programming|script|software|developer", user_inp):
            response = "Python is a popular programming language known for its simplicity and versatility. Would you like a coding example?"
            conversation_history.append((user_inp, response))
            return jsonify({"reply": response})

    # Jokes
    if re.search(r"joke|laugh|funny|make me smile|say something funny", user_inp):
        last_intent = "jokes"
        for intent in data["intents"]:
            if intent["tag"] == "jokes":
                # Avoid repetition
                available_jokes = [j for j in intent["responses"] if j not in joke_memory]
                if not available_jokes:
                    joke_memory.clear()
                    available_jokes = intent["responses"]
                response = random.choice(available_jokes)
                joke_memory.append(response)
                conversation_history.append((user_inp, response))
                # Example follow-up: ask if user wants another joke
                dialogue_state = "awaiting_joke_followup"
                response += " Want to hear another one?"
                return jsonify({"reply": response})

    if dialogue_state == "awaiting_joke_followup":
        if "yes" in user_inp or "another" in user_inp:
            for intent in data["intents"]:
                if intent["tag"] == "jokes":
                    response = random.choice(intent["responses"])
                    conversation_history.append((user_inp, response))
                    response += " Want to hear one more?"
                    return jsonify({"reply": response})
        else:
            response = "Alright! Let me know if you want to hear a joke later."
            conversation_history.append((user_inp, response))
            dialogue_state = None
            return jsonify({"reply": response})

    # Motivation
    if re.search(r"motivate|inspire|inspiring|cheer me up|sad|bored|tired", user_inp):
        last_intent = "motivation"
        for intent in data["intents"]:
            if intent["tag"] == "motivation":
                # Avoid repetition
                available_quotes = [q for q in intent["responses"] if q not in motivation_memory]
                if not available_quotes:
                    motivation_memory.clear()
                    available_quotes = intent["responses"]
                response = random.choice(available_quotes)
                motivation_memory.append(response)
                conversation_history.append((user_inp, response))
                # Example follow-up: ask if user wants more motivation
                dialogue_state = "awaiting_motivation_followup"
                response += " Want another motivational quote?"
                return jsonify({"reply": response})

    if dialogue_state == "awaiting_motivation_followup":
        if "yes" in user_inp or "another" in user_inp or "more" in user_inp:
            for intent in data["intents"]:
                if intent["tag"] == "motivation":
                    response = random.choice(intent["responses"])
                    conversation_history.append((user_inp, response))
                    response += " Want one more?"
                    return jsonify({"reply": response})
        else:
            response = "Alright! Let me know if you want more motivation later."
            conversation_history.append((user_inp, response))
            dialogue_state = None
            return jsonify({"reply": response})
    # Sentiment analysis
    sentiment = TextBlob(user_inp).sentiment.polarity

    # --- Name memory ---
    if "my name is" in user_inp or "i am" in user_inp:
        name = user_inp.split("is")[-1].strip().capitalize()
        bot_reply = f"Nice to meet you, {name}!"
        conversation_history.append((user_inp, bot_reply))
        return jsonify({"reply": bot_reply})

    if "whats my name" in user_inp or "what is my name" in user_inp:
        if name:
            bot_reply = f"Your name is {name}."
        else:
            bot_reply = "I don’t know your name yet. Say 'my name is ...'"
        conversation_history.append((user_inp, bot_reply))
        return jsonify({"reply": bot_reply})

    # --- Time/date ---
    if "time" in user_inp:
        last_intent = "time"
        return jsonify({"reply": f"The time is {datetime.datetime.now().strftime('%H:%M:%S')}"})
    if "date" in user_inp:
        last_intent = "date"
        return jsonify({"reply": f"Today’s date is {datetime.date.today()}."})

    # --- Weather with city detection ---
    if "weather" in user_inp:
        last_intent = "weather"
        # Try to extract city from input (naive regex)
        match = re.search(r"in ([a-zA-Z\s]+)", user_inp)
        city = match.group(1).strip().title() if match else (last_entity or "London")
        last_entity = city  # remember last mentioned city
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            if res.get("main"):
                temp = res["main"]["temp"]
                desc = res["weather"][0]["description"]
                return jsonify({"reply": f"The weather in {city} is {desc} with {temp}°C."})
            else:
                return jsonify({"reply": f"I couldn’t fetch the weather for {city}."})
        except:
            return jsonify({"reply": "Weather service is unavailable at the moment."})

    # --- DuckDuckGo API ---
    if re.search(r"\b(who is|what is|tell me about)\b", user_inp):
        last_intent = "knowledge"
        query = re.sub(r"\b(who is|what is|tell me about)\b", "", user_inp).strip()
        try:
            url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_redirect=1&no_html=1"
            res = requests.get(url).json()
            answer = res.get("AbstractText") or res.get("Answer") or res.get("Definition")
            related_topics = res.get("RelatedTopics")
            ddg_url = res.get("AbstractURL")
            truncated = False
            if answer:
                # Truncate long answers and always end with a clickable link
                max_len = 180
                if len(answer) > max_len:
                    short_answer = answer[:max_len].rsplit(". ", 1)[0] + "..."
                else:
                    short_answer = answer
                reply = short_answer
                if ddg_url:
                    reply += f' <a href="{ddg_url}" target="_blank">Read more</a>'
                return jsonify({"reply": reply})
            elif related_topics:
                # Try to get a related topic snippet
                related = related_topics[0]
                if isinstance(related, dict) and related.get("Text"):
                    text = related["Text"]
                    truncated = text.endswith("...") or len(text) > 180
                    if len(text) > 180:
                        short_text = text[:180].rsplit(". ", 1)[0] + "."
                    else:
                        short_text = text
                    reply = short_text
                    if truncated:
                        reply += "..."
                        dialogue_state = None
                    # Add DuckDuckGo link if available
                    if ddg_url:
                        reply += f' <a href="{ddg_url}" target="_blank">read more</a>'
                    return jsonify({"reply": reply})
            return jsonify({"reply": "Sorry, I couldn’t find information on that."})
        except Exception:
            return jsonify({"reply": "Something went wrong while searching DuckDuckGo."})

    # --- Advanced Intent Classification (placeholder for BERT/transformers) ---
    # For now, use existing ML model, but you can replace this with a transformer-based classifier
    X_test = vectorizer.transform([user_inp])
    proba = clf.predict_proba(X_test)[0]
    max_prob = max(proba)
    pred = clf.classes_[proba.argmax()]

    # --- Fallback Strategies ---
    if max_prob < 0.2:
        # Count consecutive clarifying questions
        clarifying_count = 0
        for prev_user, prev_bot in reversed(conversation_history):
            if prev_bot.startswith("Can you clarify") or "rephrase" in prev_bot or "not sure I understood" in prev_bot:
                clarifying_count += 1
            else:
                break
        if clarifying_count >= 1:
            bot_reply = "I’m still having trouble understanding. Here are some things you can ask me: 'What's the weather?', 'Tell me about Python', 'Give me a study tip.'"
        elif last_intent == "weather" and last_entity:
            bot_reply = f"Do you mean the weather in {last_entity} tomorrow?"
        elif last_intent == "study_help":
            bot_reply = "Are you asking about study tips or exam strategy?"
        else:
            # Use context to clarify
            if conversation_history:
                bot_reply = f"Can you clarify what you mean by: '{user_inp}'?'"
            else:
                bot_reply = random.choice(["Could you please rephrase?", "I’m not sure I understood that. Can you clarify?"])
        conversation_history.append((user_inp, bot_reply))
        return jsonify({"reply": bot_reply})

    # --- Normal response ---
    response = "Sorry, I didn’t understand that."
    for intent in data["intents"]:
        if intent["tag"] == pred:
            # Avoid repetition for greetings
            if pred == "greeting":
                available_greetings = [r for r in intent["responses"] if r not in greeting_memory]
                if not available_greetings:
                    greeting_memory.clear()
                    available_greetings = intent["responses"]
                response = random.choice(available_greetings)
                greeting_memory.append(response)
            else:
                response = random.choice(intent["responses"])
            break

    # Emotion detection: adjust response if negative sentiment
    if sentiment < -0.3:
        response += " (I sense you might be upset. If you want to talk about it, I’m here to listen.)"

    last_intent = pred
    conversation_history.append((user_inp, response))
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
