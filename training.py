import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Inisialisasi Lemmatizer
lemmatizer = WordNetLemmatizer()

# Mengambil data dari file JSON
data_file = open('data.json').read()
intents = json.loads(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenisasi dan pemrosesan data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenisasi setiap kata
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Menambahkan dokumen ke korpus
        documents.append((w, intent['tag']))
        # Menambahkan tag ke kelas
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize dan lowercase setiap kata, serta menghapus kata yang diabaikan
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # Menghapus duplikat dan urutkan kata
classes = sorted(list(set(classes)))  # Urutkan kelas

# Menampilkan informasi tentang data
print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Simpan kata dan kelas ke file pickle
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Persiapkan data pelatihan
training = []
output_empty = [0] * len(classes)

for doc in documents:
    # Membuat array untuk Bag of Words
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # Membuat Bag of Words
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output: 1 untuk tag saat ini, 0 untuk tag lainnya
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Mengacak data pelatihan dan mengonversinya ke dalam array NumPy
random.shuffle(training)
training = np.array(training, dtype=object)  # Gunakan dtype=object untuk menghindari error

# Membagi data pelatihan ke dalam X (input) dan Y (output)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Periksa bentuk dari train_x dan train_y
print(f"train_x shape: {np.array(train_x).shape}")
print(f"train_y shape: {np.array(train_y).shape}")

# Membangun model neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Kompilasi model dengan SGD dan learning_rate yang benar
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Melatih model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Menyimpan model dan riwayat pelatihan
model.save('model.h5')
pickle.dump(hist.history, open('history.pkl', 'wb'))

print("Model created and saved successfully.")
