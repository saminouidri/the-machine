import cv2
import tkinter as tk
from PIL import Image, ImageTk
from deepface import DeepFace as DF

class StaticImageGenderD:
    def __init__(self, master, image_path, mode):
        self.master = master
        self.image_path = image_path
        self.mode = mode
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if mode == 'race':
            self.action = 'race'
            self.analysis = 'dominant_race'
        else:
            self.action = 'gender'
            self.analysis = 'dominant_gender'

        self.setup_gui()
        self.process_image()

    def setup_gui(self):
        self.canvas = tk.Canvas(self.master, width=800, height=600)  # Adjust size as needed
        self.canvas.pack()

    def process_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("Image not found")
            return

        #mise a l'echelle de l'image
        img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_AREA)


        # Conversion de l'image en RGB pour DeepFace
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(img_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = img_rgb[y:y + h, x:x + w]
            try:
                analysis = DF.analyze(face_roi, actions=[self.action], enforce_detection=False)[0]

                if 'race' in analysis:
                    races = analysis['race']
                    sorted_races = sorted(races.items(), key=lambda item: item[1], reverse=True)[:3]
                    race_text = ', '.join([f"{race}: {percent:.2f}%" for race, percent in sorted_races])
                    cv2.putText(img_rgb, race_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

                cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if 'gender' in analysis:
                    dominant_gender = analysis[self.analysis]
                    cv2.putText(img_rgb, dominant_gender, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                                2)

            except Exception as e:
                print("Error in analysis:", e)

        # Conversion de l'image en RGB pour affichage
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # guarder une reference pour eviter la suppression par le garbage collector


