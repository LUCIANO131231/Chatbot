
import speech_recognition as sr

rec = set.Recognizer()

with sr.Microphone() as origen:
    print("Escuchando...")
    audio = rec.listen(origen)

print("Reconociendo...")
texto = sr.recognize_google(audio, language='es-ES')
print(texto)
