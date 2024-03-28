import cv2
import numpy as np
import tensorflow as tf
import time
from PIL import Image, ImageTk
import tkinter as tk

class ObjectDetection:
    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.model = tf.saved_model.load("./pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
        self.class_names = self.read_label_map("./pretrained_models/ssd_mobilenet_v2_320x320_coco17_tpu-8/mscoco_label_map.pbtxt")
        self.class_colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    @staticmethod
    def read_label_map(label_map_path):
        item_id = None
        item_name = None
        items = {}
        
        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id" in line:
                    item_id = int(line.split(":", 1)[1].strip())
                elif "display_name" in line: #elif "name" in line:
                    item_name = line.split(":", 1)[1].replace("'", "").strip()

                if item_id is not None and item_name is not None:
                    items[item_id] = item_name
                    item_id = None
                    item_name = None

        return items

    def start_detection(self):
        self.root = tk.Toplevel()
        self.root.title("Détection d'objets")
        self.root.geometry("850x850")
        self.root.config(bg="#FFFFFF")  # Définir la couleur de fond sur blanc

        # Titre général
        self.title_label = tk.Label(self.root, text="Détection d'objets", font=("Helvetica", 16, "bold"), pady=10, bg="#FFFFFF")
        self.title_label.pack()

        self.label = tk.Label(self.root)
        self.label.pack()

        self.object_label = tk.Label(self.root, text="Entrez l'objet identifié :", font=("Helvetica", 12), bg="#FFFFFF")
        self.object_label.pack()

        self.object_entry = tk.Entry(self.root)
        self.object_entry.pack()

        self.submit_button = tk.Button(self.root, text="Soumettre", command=self.submit_object, font=("Helvetica", 12), bg="#3498DB", fg="white")
        self.submit_button.pack()

        start_time = time.time()
        frame_count = 0

        while True:
            ret, img = self.video_capture.read()
            if not ret:
                break

            frame_count += 1

            img = cv2.resize(img, (800, 600))  # Redimensionner l'image
            h, w, _ = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(img, 0)

            # Prédiction avec le modèle
            resp = self.model(input_tensor)

            # Itérer sur les boîtes, les indices de classe et les scores
            for boxes, classes, scores in zip(resp['detection_boxes'].numpy(), resp['detection_classes'], resp['detection_scores'].numpy()):
                for box, cls, score in zip(boxes, classes, scores):
                    if score > 0.61:  # Utiliser uniquement les détections avec une confiance supérieure à 0.6
                        ymin = int(box[0] * h)
                        xmin = int(box[1] * w)
                        ymax = int(box[2] * h)
                        xmax = int(box[3] * w)
                                        
                        cls = int(cls)  # Convertir le tenseur en index
                        label = "{}: {:.2f}%".format(self.class_names[cls], score * 100)
                        cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, self.class_colors[cls], 1)
                        
                        X = (xmax + xmin) / 2
                        Y = (ymax + ymin) / 2
                        poslbl = "X: ({},{})".format(X, Y)
                        cv2.circle(img, (int(X) - 15, int(Y)), 1, self.class_colors[cls], 2)    
                        cv2.putText(img, poslbl, (int(X), int(Y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.class_colors[cls], 2)
                        
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), self.class_colors[cls], 4)
                    
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            self.label.config(image=img)
            self.label.image = img
            self.root.update()

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time

        print("[INFO] Temps écoulé : {:.2f} secondes".format(elapsed_time))
        print("[INFO] FPS approximatif : {:.2f}".format(fps))

        self.root.mainloop()
        self.video_capture.release()

    def submit_object(self):
        object_identified = self.object_entry.get()
        print("Objet identifié :", object_identified)
