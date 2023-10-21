import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Download the punkt tokenizer for nltk
nltk.download('punkt')

# Sample conversational data
conversations = [
    ["Hi", "Hello! How can I help you today?"],
    ["What's the weather like today?", "I'm sorry, I can't provide real-time weather information."],
    ["Tell me a joke", "Why did the chicken cross the road? To get to the other side!"],
    ["Goodbye", "Goodbye! Have a great day!"]
]

# Preprocess the data
questions, answers = zip(*conversations)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)

# Pad sequences to a fixed length
max_sequence_length = max(len(max(questions_seq,key=len)) len(max(answers_seq,key=len))
questions_seq = pad_sequences(questions_seq, maxlen=max_sequence_length, padding='post')
answers_seq = pad_sequences(answers_seq, maxlen=max_sequence_length, padding='post'))

# Create an LSTM-based Seq2Seq model
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
X = questions_seq
y = np.array([tf.keras.utils.to_categorical(answer, num_classes=vocab_size) for answer in answers_seq])

model.fit(X, y, epochs=1000, verbose=0)

# Define a function to generate responses
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    predicted = model.predict(input_seq)
    predicted_word_index = np.argmax(predicted, axis=2)[0]
    response = [tokenizer.index_word[i] for i in predicted_word_index if i != 0]
    return ' '.join(response)

# Chat with the bot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print("ChatBot:", response)
