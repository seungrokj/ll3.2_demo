import argparse

import gradio as gr
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8080/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    #required=True,
                    required=False,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.8,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

image_url_1 = "https://images.squarespace-cdn.com/content/v1/5e7c3ad4ab841d781e6be704/1630685764584-GFMZ1DKHYXZQ471L6RCB/cat-HTPBCL8.jpg"
image_url_2 = "https://oncubanews.com/en/wp-content/uploads/2016/02/Nikolis1-755x4901.jpg"
image_url_3 = "https://d2rdhxfof4qmbb.cloudfront.net/wp-content/uploads/20200326185225/iStock-987982892.jpg"

def predict(message, history):
    # Convert chat history to OpenAI format
    history_openai_format = [{
        "role": "system",
        "content": "You are a great ai assistant."
    }]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role": "assistant",
            "content": assistant
        })
    history_openai_format.append({"role": "user", "content": message})


    if message == "CAT":
        image_url = image_url_1
    elif message == "SHIP":
        image_url = image_url_2
    elif message == "PLACE":
        image_url = image_url_3

    ## Use image url in the payload
    stream = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
            ],
        }],
        model=model,
        max_tokens=1024,
        stream=True
    )

    ## Create a chat completion request and send it to the API server
    #stream = client.chat.completions.create(
    #        #model=args.model,  # Model name to use
    #    model=model,  # Model name to use
    #    messages=history_openai_format,  # Chat history
    #    temperature=args.temp,  # Temperature for text generation
    #    stream=True,  # Stream response
    #    extra_body={
    #        'repetition_penalty':
    #        1,
    #        'stop_token_ids': [
    #            int(id.strip()) for id in args.stop_token_ids.split(',')
    #            if id.strip()
    #        ] if args.stop_token_ids else []
    #    })

    #result = stream.choices[0].message.content
    #print("Chat completion output:", result)
    #yield result

    ## Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message

# Create and launch a chat interface with Gradio
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(
                """
                # AMD MI300X GPU running Llama-3.2-90B-Vision-Instruct 🌟
                """
                )
    with gr.Row():
        gr.Image(image_url_1, height=300, width=300, label="Tell me about this CAT")
        gr.Image(image_url_2, height=300, width=300, label="Tell me about this SHIP")
        gr.Image(image_url_3, height=300, width=300, label="Tell me about this PLACE")
    with gr.Column(scale=3):
        gr.ChatInterface(
                predict,
                fill_height=True,
                title="Choose one from Examples and click Submit",
                examples=["CAT", "SHIP", "PLACE"],
                )

demo.queue().launch(server_name=args.host,
                     server_port=args.port,
                     share=True)
