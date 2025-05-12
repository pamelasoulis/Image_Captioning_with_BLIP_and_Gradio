from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

### STEP 1 
#After loading the processor and the model, you need to initialize the image to be captioned. The image data needs to be loaded and pre-processed to be ready for the model.

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
img_path = "download.jfif"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')


### STEP 2
#Next, the pre-processed image is passed through the processor to generate inputs in the required format. The return_tensors argument is set to "pt" to return PyTorch tensors.
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")


### STEP 3
#You then pass these inputs into your model's generate method.
# The argument max_new_tokens=50 specifies that the model should generate a caption of up to 50 tokens in length.

#The two asterisks (**) in Python are used in function calls to unpack dictionaries and pass items in
    # the dictionary as keyword arguments to the function. **inputs is unpacking the inputs dictionary and passing its items as arguments to the model.

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)


### STEP 4
#the generated output is a sequence of tokens. To transform these tokens into human-readable text,
    # you use the decode method provided by the processor. The skip_special_tokens argument is set to
    # True to ignore special tokens in the output text.

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)






