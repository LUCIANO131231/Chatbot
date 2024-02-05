#IMPORTA LA LIBRERIA JSON, SIRVE PARA LEER EL ARCHIVO QUE SE HA CREADO

import json

#IMPORTA LA LIBRERIA PARA HACER USO DEL PROCESAMIENTO DE LENGUAJE NATURAL
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer

import pickle
import random
import numpy as np

#IMPORTAR RELACIONADO CON LA RED NEURONAL

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers.legacy import SGD

lematizador_español = SnowballStemmer('spanish')

intents = json.loads(open('intents.json', encoding='utf-8').read())

#CREAR UNA LISTA DE PALABRAS PERO QUE COMPONEN MIS PREGUNTAS, CLASES DE SALUDOS VA A RESPONDER

#LEMATIZAR PARA EVITAR SI ES MASCULINO O NO, SACA LAS VOCALES
lista_palabras = []
palabras = []
clases = []
documentos = [] #RELACIONAR CON LISTA DE PALABRAS LAS CLASES
ignorar = [',', '?']

#DEFINIENDO LA VARIABLE CON FOR
for intent in intents["intents"]:
  for pattern in intent["patterns"]: #LO QUE SE HACE ES CREAR UNA VARIABLE DENTRO DE INTENTS Y QUE RECORRE LO QUE HAY EN EL PATTERNS
    lista_palabras = nltk.tokenize.word_tokenize(pattern)
    palabras.extend(lista_palabras)
    documentos.append((lista_palabras, intent["tag"]))
    if intent["tag"] not in clases:
      clases.append(intent["tag"]) #MIENTRAS HAYA MAS CLASES, SE MEJORA EL SALUDO
palabras = [lematizador_español.stem(palabra) for palabra in palabras if palabra not in ignorar] #CADA PALABRA DENTRO DE LA LISTA, SIEMPRE Y CUANDO NO SEA UNA COMA O SIGNO DE INTERROGACION

palabras = sorted(set(palabras))
clases = sorted(set(clases))

#SIRVE PARA CONVERTIR MIS PALABRAS EN UN ARCHIVO PKL, SE USARA PARA EL ENTRENAMIENTO DE RED NEURONAL
pickle.dump(palabras, open("palabras.pkl", "wb"))
pickle.dump(clases, open("clases.pkl", "wb"))

entrenamiento = []
salida_vacia = [0] * len(clases) #UN SOLO SALUDO TAG

for documento in documentos:
  bolsa = []
  patrones_palabras = documento[0]
  patrones_palabras = [lematizador_español.stem(palabra) for palabra in patrones_palabras]

  for palabra in palabras:
    bolsa.append(1) if palabra in patrones_palabras else bolsa.append(0)
  
  salida = list(salida_vacia)
  salida[clases.index(documento[1])] = 1

  entrenamiento.append([bolsa, salida])

random.shuffle(entrenamiento) #LO QUE HARÁ ES CHOCOLATEAR, DE MANERA ALEATORIA
entrenamiento = np.array(entrenamiento)

#PARA PODER HACER LAS PREGUNTAS
ent_x = list(entrenamiento[:,0])
ent_y = list(entrenamiento[:,1])

#CREACION DE LA RED NEURONAL
#SE TIENE 128 NEURONAS, PERO SE RECIBEN LAS PALABRAS EN 0 Y 1

modelo = Sequential()
modelo.add(Dense(128, input_shape=(len(ent_x[0]),), activation='relu')) #CAPA DE ENTRADA, CAPA DENSA, PRIMERA CAPA DE LA RED NEURONAL
modelo.add(Dropout(0.5)) #LO QUE SIGUE DE LA CAPA
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(ent_y[0]), activation='softmax'))

#PARA OPTIMIZAR LA TASA DE APRENDIZAJE
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

proc_ent = modelo.fit(np.array(ent_x), np.array(ent_y), epochs=200, batch_size=5, verbose=1)

modelo.save("chatbot.h5",proc_ent)
