from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []


while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

# Tokens in NLP are individual units or elements that text or sentences are divided into.
# Tokenization or vectorization is the process of converting tokens into numerical representations.
# In NLP tasks, you often use the encode_plus method from the tokenizer object to perform tokenization
    # and vectorization. Let's encode your inputs (prompt & chat history) as tokens so that you may pass
    # them to the model.
    
    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs)

# You may decode the output using tokenizer.decode(). This is known as "detokenization" or "reconstruction". 
# It is the process of combining or merging individual tokens back into their original form, to 
    # reconstruct the original text or sentence.
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)


    









