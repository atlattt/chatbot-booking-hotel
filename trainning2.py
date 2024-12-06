import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from sklearn.model_selection import train_test_split
import pickle

sys.stdout.reconfigure(encoding='utf-8')  # Đặt mã hóa UTF-8 cho đầu ra
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load từ viết tắt từ file JSON
with open('data/abbreviations.json', 'r', encoding='utf-8') as f:
    abbreviations = json.load(f)

# Hàm thay thế từ viết tắt trong câu
def expand_abbreviations(text):
    words = text.split()
    return " ".join([abbreviations.get(word, word) for word in words])
words=[]
classes=[]
documents=[]
ignore_words = ["?", "!"]
for intent in intents['intents']:
    intent['pattern'] = [expand_abbreviations(pattern) for pattern in intent['pattern']]
    for pattern in intent['pattern']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
print(words)
classes=sorted(list(set(classes)))
print(classes)
pickle.dump(words, open("data/texts.pkl", "wb"))
pickle.dump(classes, open("data/labels.pkl", "wb"))
# Tạo vector cho tất cả các pattern
training_data = []
output_empty = [0] * len(classes)
#print(list(output_empty))
#print(classes)
for doc in documents:
     # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training_data.append([bag, output_row])
random.shuffle(training_data)
train_x = np.array([item[0] for item in training_data])
train_y = np.array([item[1] for item in training_data])

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Dừng sớm nếu val_loss không giảm trong 10 epochs
sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=200, batch_size=5, callbacks=[early_stopping],verbose=1)

model.save('model.keras')


