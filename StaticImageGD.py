import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

class StaticImageGD:
    def __init__(self, master, image_path):
        self.master = master
        self.image_path = image_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.setup_gui()
        self.process_image()

    def setup_gui(self):
        self.canvas = tk.Canvas(self.master, width=800, height=600)
        self.canvas.pack()

    def process_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            print("Image not found")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = self.detector(img_rgb, 1)

        for rect in rects:
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            sp = self.predictor(img_rgb, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            img_cropped = self.isolate_nose_bridge(landmarks, img)
            edges_detected, edges_image = self.glass_edge_detection(img_cropped)

            if edges_detected:
                cv2.putText(img, "Glasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, "No Glasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def isolate_nose_bridge(self, landmarks, img):
        nose_bridge = landmarks[27:36]
        x_min = min(nose_bridge[:, 0])
        x_max = max(nose_bridge[:, 0])
        y_min = min(nose_bridge[:, 1])
        y_max = max(nose_bridge[:, 1])
        img_pil = Image.fromarray(img)
        return img_pil.crop((x_min, y_min, x_max, y_max))

    def glass_edge_detection(self, img_cropped):
        img_blur = cv2.GaussianBlur(np.array(img_cropped), (3, 3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        edges_center = edges.T[int(len(edges.T) / 2)]
        return 255 in edges_center, edges
