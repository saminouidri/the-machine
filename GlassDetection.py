import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import threading
import tkinter as tk

#source du classificateur : https://medium.com/@siddh30/glasses-detection-opencv-dlib-bf4cd50856da
class GlassDetection:
    def __init__(self, master, debug=False):
        self.master = master
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.debug = debug

        self.setup_gui()
        self.start_video_thread()

    def setup_gui(self):
        self.canvas = tk.Canvas(self.master, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

    def start_video_thread(self):
        self.thread = threading.Thread(target=self.detect_attribute)
        self.thread.daemon = True
        self.thread.start()

    def detect_attribute(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rects = self.detector(frame_rgb, 1)
                for rect in rects:
                    landmarks = self.get_landmarks(frame_rgb, rect)
                    self.process_frame(frame, rect, landmarks)
                self.update_gui(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()

    def process_frame(self, frame, rect, landmarks):
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if landmarks.size > 0:
            img_cropped = self.isolate_nose_bridge(landmarks, frame)
            if img_cropped:
                edges_detected, edges_image = self.glass_edge_detection(img_cropped)
                if edges_detected:
                    if self.debug:
                        self.display_edges_window(edges_image)
                    cv2.putText(frame, "Glasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    if self.debug:
                        self.display_edges_window(edges_image)
                    cv2.putText(frame, "No Glasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def get_landmarks(self, img, rect):
        sp = self.predictor(img, rect)
        return np.array([[p.x, p.y] for p in sp.parts()])

    def isolate_nose_bridge(self, landmarks, frame):
        nose_bridge = landmarks[27:36]
        x_min = min(nose_bridge[:, 0])
        x_max = max(nose_bridge[:, 0])
        y_min = min(nose_bridge[:, 1])
        y_max = max(nose_bridge[:, 1])
        img_pil = Image.fromarray(frame)
        return img_pil.crop((x_min, y_min, x_max, y_max))

    def glass_edge_detection(self, img_cropped):
        img_blur = cv2.GaussianBlur(np.array(img_cropped), (3, 3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        edges_center = edges.T[int(len(edges.T) / 2)]
        return 255 in edges_center, edges

    def display_edges_window(self, edges_image):
        edges_image_pil = Image.fromarray(edges_image)
        edges_image_tk = ImageTk.PhotoImage(image=edges_image_pil)

        if not hasattr(self, 'edges_window') or not self.edges_window.winfo_exists():
            self.edges_window = tk.Toplevel(self.master)
            self.edges_window.title("Edges Detected")
            self.edges_label = tk.Label(self.edges_window)
            self.edges_label.pack()

        self.edges_label.config(image=edges_image_tk)
        self.edges_label.image = edges_image_tk  # pour eviter la suppression par le garbage collector

    def update_gui(self, frame):
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk  # pour eviter la suppression par le garbage collector
