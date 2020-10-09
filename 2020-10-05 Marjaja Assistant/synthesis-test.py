# https://habr.com/ru/post/515128/

import datetime
today = datetime.datetime.today()
 
import pyttsx3

# C:\python38\python.exe 'c:\topcoder\topcoder\2020-10-05 Marjaja Assistant\synthesis-test.py'

tts = pyttsx3.init()
rate = tts.getProperty('rate') #Скорость произношения
tts.setProperty('rate', rate-10)

volume = tts.getProperty('volume') #Громкость голоса
tts.setProperty('volume', volume+0.9)

voices = tts.getProperty('voices')

# Задать голос по умолчанию
tts.setProperty('voice', 'en') 

# Попробовать установить предпочтительный голос
for voice in voices:
    if voice.name == 'Aleksandr':
        tts.setProperty('voice', voice.id)

text = 'Здравствуйте, меня зовут Марджаджа, я ваш голосовой помощник. Что вы хотите спросить у меня?'
tts.say(text)
tts.runAndWait()

"""
tts.stop()

path_to_save = "c:\\topcoder\\topcoder\\2020-10-05 Marjaja Assistant\\audio.mp3"
tts.save_to_file(text, path_to_save)
tts.runAndWait()

quit()
"""

import difflib

def similarity(s1, s2):
  normalized1 = s1.lower()
  normalized2 = s2.lower()
  matcher = difflib.SequenceMatcher(None, normalized1, normalized2)
  return matcher.ratio()

import speech_recognition as sr
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

def record_volume():
    r = sr.Recognizer()
    with sr.Microphone(device_index = 1) as source:
        print(today.strftime("%H:%M:%S"))
        print('Настраиваюсь.')
        r.adjust_for_ambient_noise(source, duration=0.5) #настройка посторонних шумов
        print('Слушаю...')
        audio = r.listen(source, phrase_time_limit=5)
    print('Услышала.')
    try:
        query = r.recognize_google(audio, language = 'ru-RU')
        text = query.lower()
        print(f'Вы сказали: {query.lower()}')
        tts.say("Я услышал вас. Вы сказали." + text + ". Отвечаю.")
        tts.runAndWait()

        if (similarity(text, "спой розенбаума") > 0.7):
            text = 'Налетела грусть. Что ж, пойду пройдусь. Мне её делить не с кем. И зеленью аллей. В пухе тополей. Я иду землёй Невской.'
            tts.say(text)
            tts.runAndWait()
        if (similarity(text, "спой песню") > 0.7):
            text = 'Марджанжа, Марджанжа, где же ты? Где? Волны ласкают усталые скалы. Марджанжа, Марджанжа, где же ты? Где? Только блики на воде.'
            tts.say(text)
            tts.runAndWait()
        if (similarity(text, "поприветствуй анатолия") > 0.7):
            text = 'Анатолий, Анатолий, Ты от муки, ты от боли, Ты от скуки, ты от слёз, Радость в душу мне принёс. Толька, Толька, Толька, Толька, А не Лёшка и не Колька'
            tts.say(text)
            tts.runAndWait()            
        if (similarity(text, "который час") > 0.7):
            text = 'Текущее время ' + today.strftime("%H") + ' часов ' + today.strftime("%M") + ' минут. Вынужден вам напомнить. Убытие с рабочего места до семнадцати тридцати будет воспринято как личное оскорбление начальника. Призываю вас соблюдать регламент служебного времени!'
            tts.say(text)
            tts.runAndWait()                  

    except:
        print('Error')

while True:
    record_volume()
    break