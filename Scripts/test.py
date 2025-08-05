import os
from dotenv import load_dotenv
from google import genai

#Load API key
load_dotenv()  
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# One-off text generation
resp = client.models.generate_content(
    model="gemini-2.5-flash",    # or whichever Gemini model you’ve access to
    contents="Say hi!"
)
print("Generate content response:", resp.text)

# Chat-style interaction
chat = client.chats.create(model="gemini-2.5-flash")
reply = chat.send_message("Hello, how are you?")
print("Chat response:", reply.text)
