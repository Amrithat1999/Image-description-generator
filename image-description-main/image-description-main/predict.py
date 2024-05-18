

import numpy as np
from PIL import Image
from pickle import load
from keras.applications.xception import preprocess_input
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import cv2


def extract_features(image, x_interpreter):

    if type(image)==type(""):
        image = Image.open(image)
    else:
        image = Image.open(image.stream)
    img = image.resize((299,299))
    img = img.convert('RGB')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    img = np.array([img])
    x_interpreter.set_tensor(ix_index, img)
    x_interpreter.invoke()
    feature = x_interpreter.get_tensor(ox_index)
    del image
    return feature

def word_for_id(integer):
    global tokenizer
    for word, index in tokenizer.word_index.items():
         if index == integer:
             return word
    return None

def generate_desc_lite(photo,interpreter):

    global tokenizer
    global max_length
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences(
            [sequence], maxlen=max_length).astype(np.float32)
        interpreter.set_tensor(i_index_1, sequence)
        interpreter.set_tensor(i_index_2, photo)
        interpreter.invoke()
        pred = interpreter.get_tensor(o_index)
        pred = np.argmax(pred)
        word = word_for_id(pred)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def predict(image):
    photo = extract_features(image, x_interpreter)
    description = generate_desc_lite(photo,interpreter)
    return description


max_length = 47
tokenizer = load(open("files/tokenizer.p","rb"))


interpreter = tf.lite.Interpreter("files/model-110.tflite")
interpreter.allocate_tensors()
i_details = interpreter.get_input_details()
o_details = interpreter.get_output_details()
i_index_1 = i_details[0]['index']
i_index_2 = i_details[1]['index']
o_index = o_details[0]['index']


x_interpreter = tf.lite.Interpreter("files/xception.tflite")
x_interpreter.allocate_tensors()
ix_details = x_interpreter.get_input_details()
ox_details = x_interpreter.get_output_details()
ix_index = ix_details[0]['index']
ox_index = ox_details[0]['index']


import nltk
from nltk.translate.bleu_score import corpus_bleu

# Example test dataset (replace with your actual test dataset)
test_data = [(image_path, reference_caption)]

# Function to tokenize captions
def tokenize_captions(captions):
    tokenized_captions = []
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokenized_captions.append(tokens)
    return tokenized_captions

# Generate captions and tokenize
predicted_captions = []
reference_captions = []
for image_path, reference_caption in test_data:
    predicted_caption = predict(image_path)
    predicted_captions.append(predicted_caption)
    reference_captions.append(reference_caption)

# Tokenize captions
predicted_captions = tokenize_captions(predicted_captions)
reference_captions = [tokenize_captions([caption]) for caption in reference_captions]

# Compute BLEU score (other metrics can be computed similarly)
bleu_score = corpus_bleu(reference_captions, predicted_captions)

print("BLEU Score:", bleu_score)


