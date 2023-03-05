import gradio as gr
import openai
import requests
import json
import os
import dotenv
from dotenv import load_dotenv


load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

    
messages = [{"role": "system", "content": 'You are Steve Jobs. Respond to all input in 25 words or less.'}]

# Set up the API endpoint URL and headers
url = f"https://api.elevenlabs.io/v1/text-to-speech/{os.environ.get('voice_id')}/stream"
headers = {
    "accept": "*/*",
    "xi-api-key": os.environ.get('elevenlabs_api_key'),
    "Content-Type": "application/json",
}

# Define a function to handle the Gradio input and generate the response
def transcribe(audio):
    global messages

    # Use OpenAI to transcribe the user's audio input
    # API call 1
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Append the user's message to the message history
    messages.append({"role": "user", "content": transcript["text"]})

    # Generate a response using OpenAI's chat API
    #API call 2
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # Extract the system message from the API response and append it to the message history
    system_message = response["choices"][0]["message"]
    messages.append(system_message)
    
    
    #API Call 3
    # Use the voice synthesis API to generate an audio response from the system message
    data = {
        "text": system_message["content"],
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    # Save the audio response to a file
    if response.ok:
        with open("output.wav", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        print(f"Error: {response.status_code} - {response.reason}")
        
    # IPython.display.display(IPython.display.Audio('output.wav'))

    # Generate a chat transcript for display in the Gradio UI
    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript,'output.wav'



# css = """
#       #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
#       #header {text-align: center;}
#         }
#         """

# with gr.Blocks(css=css) as ui:
    
    
#     with gr.Column(elem_id="col-container"):
#         gr.Markdown("""## Talk to AI Steve Jobs: Audio-to-Text+Audio generation
#                     Powered by ChatGPT + Whisper + ElevenLabs + HuggingFace <br>
#                     <br>
#                     """,
#                     elem_id="header")

# Define the Gradio UI interface
# ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text")
ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), title='Talk to AI Steve Jobs', outputs=['text','audio'],description = """Click on Record from microphone and start speaking, and when you're done, click on Stop Recording. Then click on Submit. The AI Steve Jobs will then answer your question. You can then continue to ask follow-up questions by clicking on Clear, and then using Record from microphone -> Stop Recording -> Submit  AI Steve Jobs will also remember the previous questions and answers.""")
ui.launch(debug=True)
