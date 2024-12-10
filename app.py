import nltk
# Unduh hanya yang diperlukan
nltk.download('punkt')  # Untuk tokenisasi
nltk.download('wordnet')  # Untuk lemmatizer

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random


# Memuat model dan data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Fungsi untuk membersihkan kalimat
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenisasi kalimat
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize
    return sentence_words

# Fungsi untuk membuat bag of words dari kalimat
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Fungsi untuk memprediksi kelas berdasarkan kalimat
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)  # Urutkan berdasarkan probabilitas
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Fungsi untuk mendapatkan respons berdasarkan kelas yang diprediksi
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

# Fungsi utama chatbot untuk mengembalikan respons
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Flask app untuk antarmuka web
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')  # Mengambil pesan dari user
    return chatbot_response(userText)  # Mengembalikan respons chatbot

if __name__ == "__main__":
    app.run(debug=True)  # Menjalankan server Flask dalam mode debug untuk deteksi kesalahan lebih cepat
