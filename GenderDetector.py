import os

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
from deepface import DeepFace as DF

# source du classificateur : https://github.com/manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface/tree/main
# inspiré du même exemple

class GenderDetector:
    def __init__(self, master, mode):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.running = True  # flag pour savoir si le thread doit continuer a tourner
        self.stop_event = threading.Event()
        if mode == 'race':
            self.action = 'race'
            self.analysis = 'dominant_race'
        else:
            self.action = 'gender'
            self.analysis = 'dominant_gender'


        # Setup GUI
        self.setup_gui()

        # demarrage du traitement video dans un thread separe pour garder l'interface utilisateur reponsive (n'as pas l'air de trop fonctionner)
        self.start_video_thread()

    def setup_gui(self):
        self.canvas = tk.Canvas(self.master, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.quit_button = tk.Button(self.master, text="Quit", command=self.on_close)
        self.quit_button.pack(side=tk.BOTTOM)
        self.canvas.pack()

    def start_video_thread(self):
        self.thread = threading.Thread(target=self.detectAttribute)
        self.thread.daemon = True
        self.thread.start()

    def detectAttribute(self):
        while not self.stop_event.is_set():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()

    def process_frame(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                analysis = DF.analyze(face_roi, actions=[self.action], enforce_detection=False)[0]

                if 'race' in analysis:
                    races = analysis['race']
                    # source du tri : https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
                    sorted_races = sorted(races.items(), key=lambda item: item[1], reverse=True)[:3]
                    race_text = ', '.join([f"{race}: {percent:.2f}%" for race, percent in sorted_races])
                    # cv2 n'a pas de retour a la ligne, donc on split le texte pour l'afficher sur plusieurs lignes
                    first_race = race_text.split(',')[0]
                    second_race = race_text.split(',')[1]
                    third_race = race_text.split(',')[2]
                    cv2.putText(frame, first_race, (x + w, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, second_race, (x + w, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame, third_race, (x + w, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if 'gender' in analysis:
                    dominant_gender = analysis[self.analysis]
                    cv2.putText(frame, dominant_gender, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 0, 0), 2)

            except Exception as e:
                    print("Error in analysis:", e)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.update_gui(imgtk)

    def update_gui(self, imgtk):
        if self.running:  # Check if the thread should still be running
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk


    #Malgré la tentative de fermer le thread gentiment avec un join, il arrive que ce dernier ne se ferme pas
    #On force donc la fermeture du thread et de la fenetre apres 5 secondes si le thread ne s'est pas ferme
    def on_close(self):
        def force_exit():
            print("Forced exit due to timeout.")
            os._exit(1)

        timer = threading.Timer(5, force_exit)
        timer.start()

        try:
            self.running = False
            self.stop_event.set()
            self.thread.join(timeout=5)
            self.cap.release()
        finally:
            timer.cancel()
            self.master.destroy()

