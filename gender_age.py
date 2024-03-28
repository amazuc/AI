import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk

class GenderAge:
    # Définir MODEL_MEAN_VALUES comme un attribut de classe
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(self, video_capture):
        self.cap = video_capture
        self.root = None  # Garder une référence à la fenêtre
        self.age_estimated = None

    def start_detection(self):
        # Charger les réseaux
        faceProto, faceModel = "opencv_face_detector.pbtxt", "opencv_face_detector_uint8.pb"
        ageProto, ageModel = "age_deploy.prototxt", "age_net.caffemodel"
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.ageNet = cv.dnn.readNet(ageProto, ageModel)
        self.faceNet = cv.dnn.readNet(faceModel, faceProto)
        self.padding = 20

        # Créer la fenêtre Tkinter pour afficher la vidéo
        self.root = tk.Toplevel()
        self.root.title("Détection de visages et estimation d'âge")
        self.root.geometry("850x850")  # Ajuster la taille de la fenêtre
        self.root.config(bg="#FFFFFF")  # Définir la couleur de fond sur blanc

        # Titre général
        self.title_label = tk.Label(self.root, text="Détection de visages et estimation d'âge", font=("Helvetica", 16, "bold"), pady=10, bg="#FFFFFF")
        self.title_label.pack()

        # Créer un label pour afficher la vidéo
        self.label = tk.Label(self.root, bg="#FFFFFF")
        self.label.pack()

        # Créer le formulaire
        self.age_label = tk.Label(self.root, text="Âge réel :", font=("Helvetica", 12), bg="#FFFFFF")
        self.age_label.pack()

        self.age_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.age_entry.pack()

        self.submit_button = tk.Button(self.root, text="Envoyer", command=self.submit_age, font=("Helvetica", 12), bg="#3498DB", fg="white")
        self.submit_button.pack()

        def getFaceBox(net, frame, conf_threshold=0.7):
            frameHeight, frameWidth = frame.shape[:2]
            blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1, y1, x2, y2 = int(detections[0, 0, i, 3] * frameWidth), int(
                        detections[0, 0, i, 4] * frameHeight), int(detections[0, 0, i, 5] * frameWidth), int(
                        detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frame, bboxes

        # Boucle pour capturer et afficher la vidéo
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv.resize(frame, (800, 600)) 
            frameFace, bboxes = getFaceBox(self.faceNet, frame)
            if not bboxes:
                print("Aucun visage détecté, Vérification du frame suivant")
                continue

            for bbox in bboxes:
                face = frame[max(0, bbox[1] - self.padding):min(bbox[3] + self.padding, frame.shape[0] - 1),
                    max(0, bbox[0] - self.padding):min(bbox[2] + self.padding, frame.shape[1] - 1)]
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                self.ageNet.setInput(blob)
                agePreds = self.ageNet.forward()
                age = self.ageList[agePreds[0].argmax()]
                confidence = agePreds[0].max()

                print("Âge : {}, confiance = {:.3f}".format(age, confidence))

                if self.age_estimated is None or confidence > 0.85:
                    self.age_estimated = age

                cv.putText(frameFace, "Age : {}".format(self.age_estimated), (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                        cv.LINE_AA)

            # Convertir le cadre OpenCV en format Tkinter
            frameFace = cv.cvtColor(frameFace, cv.COLOR_BGR2RGB)
            frameFace = Image.fromarray(frameFace)
            frameFace = ImageTk.PhotoImage(image=frameFace)

            # Afficher la vidéo dans la fenêtre Tkinter
            self.label.config(image=frameFace)
            self.label.image = frameFace
            self.root.update()

        # Libérer la capture vidéo après avoir terminé
        self.cap.release()

    def submit_age(self):
        age = self.age_entry.get()
        print("Âge réel : ", age)

    def stop_mouse(self):
        self.root.destroy()

