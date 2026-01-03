# src/joke_generator.py

import pyttsx3
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 170)

# Emotions considered "low" to trigger cheering
LOW_EMOTIONS = ["Sad", "Angry", "Fear", "Disgust"]

def generate_joke():
    prompt = "Tell a short, friendly joke to cheer someone up."
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=500,
            top_p=1,
            reasoning_effort="medium",
            stream=False  # Set True for streaming chunks
        )

        joke_text = completion.choices[0].message.content.strip()
        if not joke_text:
            joke_text = "Hey! Everything will be okay 😊"

        return joke_text

    except Exception as e:
        print("Groq API error:", e)
        return "Cheer up! You're doing great 😊"

def cheer_user(emotion):
    """Generate and speak a joke if emotion is low."""
    if emotion not in LOW_EMOTIONS:
        return

    joke = generate_joke()
    print("Joke:", joke)
    engine.say(joke)
    engine.runAndWait()
