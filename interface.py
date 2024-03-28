import tkinter as tk
import cv2 as cv
from PIL import ImageTk, Image
from gender_age import GenderAge
from aiMouse import VirtualMouse  # Importer la classe VirtualMouse
from object_detection import ObjectDetection

class MenuApp:
    def __init__(self, master):
        self.master = master
        master.title("Menu")

        # Charger le logo et le redimensionner
        original_logo = Image.open("logo.png")
        original_logo = original_logo.resize((100, 100))
        self.logo_img = ImageTk.PhotoImage(original_logo)

        # Créer un label pour afficher le logo
        self.logo_label = tk.Label(master, image=self.logo_img)
        self.logo_label.pack(side="left", padx=10, pady=10, anchor="nw")

        # Titre général
        self.title_label = tk.Label(master, text="Bienvenue dans l'application de détection", font=("Helvetica", 16, "bold"), pady=10)
        self.title_label.pack(anchor="w")

        # Description de la souris virtuelle
        self.mouse_desc_label = tk.Label(master, text="Activer/désactiver la souris virtuelle", font=("Helvetica", 12),  pady=5)
        self.mouse_desc_label.pack()

        # Bouton pour activer/désactiver la souris virtuelle
        self.mouse_button = tk.Button(master, text="Activer", command=self.toggle_virtual_mouse, font=("Helvetica", 10, "bold"), bg="#008000", fg="white")
        self.mouse_button.pack()

        # Description de la détection de visage
        self.face_desc_label = tk.Label(master, text="Démarrer la détection de visage", font=("Helvetica", 12), pady=5)
        self.face_desc_label.pack()

        # Bouton de démarrage de la détection de visage
        self.face_button = tk.Button(master, text="Démarrer", command=self.start_face_detection, font=("Helvetica", 10, "bold"), bg="#3498DB", fg="white")
        self.face_button.pack()

        # Description de la détection d'objet
        self.obj_desc_label = tk.Label(master, text="Activer la détection d'objet", font=("Helvetica", 12), pady=5)
        self.obj_desc_label.pack()

        # Bouton d'activation de la détection d'objet
        self.obj_button = tk.Button(master, text="Démarrer", command=self.start_obj_detection, font=("Helvetica", 10, "bold"), bg="#3498DB", fg="white")
        self.obj_button.pack()

        # Initialisation de la capture vidéo
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)  # Largeur de la caméra
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)  # Hauteur de la caméra

        # Créer une instance de chaque classe
        self.gender_age_instance = GenderAge(self.cap)
        self.virtual_mouse_instance = VirtualMouse(self.cap)
        self.obj_detection_instance = ObjectDetection(self.cap)

        # Variable pour le suivi de l'état de la souris virtuelle
        self.virtual_mouse_enabled = False

    def start_face_detection(self):
        self.gender_age_instance.start_detection()

    def toggle_virtual_mouse(self):
        if self.virtual_mouse_enabled:
            self.mouse_button.config(text="Activer", bg="#008000", fg="white")  # Modifier le texte et la couleur du bouton
            self.virtual_mouse_enabled = False
            self.virtual_mouse_instance.stop_mouse()
        else:
            self.mouse_button.config(text="Désactiver", bg="#FF5733", fg="white")  # Modifier le texte et la couleur du bouton
            self.virtual_mouse_enabled = True
            self.virtual_mouse_instance.start_mouse()
            

    def start_obj_detection(self):
        self.obj_detection_instance.start_detection()

def main():
    root = tk.Tk()
    app = MenuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
