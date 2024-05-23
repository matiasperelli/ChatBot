import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.api.models import Sequential 
from keras.api.layers import Dense,Dropout
from keras.api.optimizers import SGD
from keras.api.optimizers.schedules import ExponentialDecay
import random

data_file = open('intents_spanish.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
#recorre cada intencion y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokeniza las palabras en cada patron y las agrega a la lista de palabras
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #agraga el par (patron, etiqueta) a la lista de documentos
        documents.append((w,intent['tag']))
        # si la etiqueta no esta en la lista de clases, la agrega
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lematiza las palabras y las convierte en minusculas, excluyendo las palabras ignoradas
words = [lemmatizer.lemmatize(w.lower())for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# guarda las listas de palabras y clases en archivo pickle
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

# crea el conjunto de entrenamiento 
for doc in documents:
    Bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        # crea una bolsa de palabras binaria para cada patron
        Bag.append(1) if word in pattern_words else Bag.append(0)
    output_row = list(output_empty)
   
    # crea un vector de salida con un 1 en la posicion correspondiente a la etiqueta 
    output_row[classes.index(doc[1])] = 1
    training.append([Bag, output_row])

    # mezcla aletoriamente el conjunto de entrenamiento
    random.shuffle(training)

    # divide el conjunto de entrenamiento en caracteristicas (train_x) y etiquetas (train_y)
    train_x = [row[0] for row in training]
    train_y = [row[1] for row in training]

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # crea el modelo de red neuronal 
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# configura el optimizador con una taza de aprendizaje exponencialmente decreciente
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# entrena el modelo con el conjunto de entrenamiento
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
