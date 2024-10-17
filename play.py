import base64
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Function to generate audio response
def generate_audio_response(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=messages
    )
    return completion.choices[0]

# Function to decode and save audio
def save_audio(audio_data, filename):
    wav_bytes = base64.b64decode(audio_data)
    with open(filename, "wb") as f:
        f.write(wav_bytes)

# Initialize conversation history
conversation_history = []

# Streamlit UI
st.title("Multi-turn Conversation Bot")

# User input
user_input = st.text_input("You: ", "")

if user_input:
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Generate response
    response = generate_audio_response(conversation_history)

    # Add assistant message to conversation history
    conversation_history.append(response.message)

    # Save and play audio
    audio_filename = f"{response.message.audio.id}.wav"
    save_audio(response.message.audio.data, audio_filename)
    st.audio(audio_filename, format="audio/wav", autoplay=True)

    # Display transcript
    st.write("Assistant: ", response.message.audio.transcript)

# Display conversation history
st.subheader("Conversation History")
for message in conversation_history:
    if isinstance(message, dict):  # User message
        role = "You"
        content = message["content"]
    else:  # Assistant message
        role = "Assistant"
        content = message.audio.transcript
    st.write(f"{role}: {content}")