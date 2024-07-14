import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk, Image

from GenderDetector import GenderDetector
from GlassDetection import GlassDetection
from StaticImageGD import StaticImageGD
from StaticImageGenderD import StaticImageGenderD


class MainApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Main frame
        self.main_frame = tk.Frame(window, padx=20, pady=20)
        self.main_frame.pack(padx=10, pady=10)

        # Image for logo
        self.image = Image.open("logo.png")  # Update this path
        self.photo = ImageTk.PhotoImage(self.image)
        self.image_label = tk.Label(self.main_frame, image=self.photo)
        self.image_label.pack(pady=(0, 20))  # Add some vertical space below the image

        # Label for dropdown
        self.text_label = tk.Label(self.main_frame, text="Select a detection mode:", font=("Arial", 14))
        self.text_label.pack(pady=(0, 10))

        # Dropdown for choosing detection type
        self.feature = tk.StringVar(window)
        self.features = ['Detection de lunettes', 'Estimation d\'origine', 'Detection de genre']
        self.feature.set(self.features[0])  # default value
        self.dropdown = ttk.Combobox(window, textvariable=self.feature, values=self.features, state="readonly")
        self.dropdown.pack()

        # Button to launch detection
        self.launch_button = tk.Button(window, text="Lancer", command=self.on_feature_select)
        self.launch_button.pack()

        # Checkbox for "Single Image" mode
        self.single_image_mode = tk.BooleanVar()
        self.single_image_checkbox = tk.Checkbutton(window, text="Single image mode", variable=self.single_image_mode)
        self.single_image_checkbox.pack()

        # Button to choose an image file
        self.choose_file_button = tk.Button(window, text="Choose Image", command=self.choose_file)
        self.choose_file_button.pack()

        # Initialize the image path variable
        self.image_path = None

    def choose_file(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            print("Selected:", self.image_path)

    def on_feature_select(self):
        selected_feature = self.feature.get()
        single_mode = self.single_image_mode.get()

        if selected_feature == 'Detection de genre':
            self.feature_window = tk.Toplevel(self.window)
            if single_mode and self.image_path:
                StaticImageGenderD(self.feature_window,  self.image_path, "gender")
            else:
                GenderDetector(self.feature_window, "gender")
        elif selected_feature == 'Estimation d\'origine':
            self.feature_window = tk.Toplevel(self.window)
            if single_mode and self.image_path:
                StaticImageGenderD(self.feature_window,  self.image_path, "race")
            else:
                GenderDetector(self.feature_window, "race")
        elif selected_feature == 'Detection de lunettes':
            self.glasses_window = tk.Toplevel(self.window)
            if single_mode and self.image_path:
                StaticImageGD(self.glasses_window, self.image_path)
            else:
                GlassDetection(self.glasses_window, True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root, "Tkinter and OpenCV")
    root.mainloop()
