import sys
from keras.models import load_model
import nltk
import json
#nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Load the model from disk
import re
import string
import pickle
import numpy as np
import random
import os


# Load từ viết tắt từ file JSON
with open('data/abbreviations.json', 'r', encoding='utf-8') as f:
    abbreviations = json.load(f)

# Hàm thay thế từ viết tắt trong câu
def expand_abbreviations(text):
    words = text.split()
    return " ".join([abbreviations.get(word, word) for word in words])


url_pattern = re.compile(r'http\S+')
def clean_up_sentence(sentence):

    # expand abbreviations
    sentence =expand_abbreviations(sentence)


    # ignore special characters
    sentence_trans = sentence.translate(str.maketrans('', '', string.punctuation))
    # ignore link
    sentence_trans = url_pattern.sub(r'', sentence_trans)
    # ignore emotions

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence_trans)

    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    print(sentence_words)

    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                print(w)
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model,words,classes):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    # Kiểm tra nếu ints rỗng
    if not ints:
        return "Xin lỗi, tôi không hiểu ý của bạn. Vui lòng thử lại với câu hỏi khác."
    
    # Truy cập intent đầu tiên
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = None  # Khởi tạo giá trị mặc định cho result

    # Duyệt qua danh sách intents để tìm phản hồi phù hợp
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    # Xử lý khi không tìm thấy tag
    if result is None:
        return f"Không tìm thấy phản hồi cho intent: {tag}"
    
    return result


def chatbot_response(msg):
    model = load_model('model.keras')
    intents = json.loads(open('data/intents.json',encoding="utf8").read())
    words = pickle.load(open('data/texts.pkl', 'rb'))
    classes = pickle.load(open('data/labels.pkl', 'rb'))
    ints = predict_class(msg, model,words,classes)
    print(ints)
    res=getResponse(ints, intents)
    return res
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    while True:
        msg = input("You: ")
        print(chatbot_response(msg))
        if msg == 'exit':
            break
    