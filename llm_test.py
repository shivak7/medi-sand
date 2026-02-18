import openai

# Configure the client to point to your local llama.cpp server
client = openai.OpenAI(
    base_url="http://viropa:8001/v1",  # Adjust URL and port if necessary
    api_key="sk-no-key-required"         # Dummy key
)

# Use the chat completions API as you would with OpenAI's service
completion = client.chat.completions.create(
    model="google/gemma3_4b", # The model name is not strictly necessary for llama-cpp-python server
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a limerick about python exceptions"}
    ],
    stream=False, # Set to True for streaming responses
)

# Print the response
print(completion.choices[0].message.content)
