import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk

class VirtualMouse:
    def __init__(self, video_capture):
        self.root = None
        self.cap = video_capture
        self.hand_detector = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()

    def start_mouse(self):
        self.root = tk.Toplevel()
        self.root.title("Souris Virtuelle")
        self.root.geometry("600x500")  # Ajuster la taille de la fenêtre
        self.root.withdraw()

        self.label = tk.Label(self.root)
        self.label.pack()

        index_y = 0  # Initialisation de index_y en dehors de la boucle while

        while True:
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.hand_detector.process(rgb_frame)
            hands = output.multi_hand_landmarks
            if hands:
                for hand in hands:
                    self.drawing_utils.draw_landmarks(frame, hand)
                    landmarks = hand.landmark
                    for id, landmark in enumerate(landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        if id == 8:
                            cv2.circle(frame, (x, y), radius=10, color=(0, 255, 255))
                            index_x = self.screen_width / frame_height * x
                            index_y = self.screen_height / frame_width * y
                            pyautogui.moveTo(index_x, index_y)
                        if id == 4:
                            cv2.circle(frame, (x, y), radius=10, color=(0, 255, 255))
                            thumb_x = self.screen_width / frame_height * x
                            thumb_y = self.screen_height / frame_width * y
                            print('outside', abs(index_y - thumb_y))
                            if abs(index_y - thumb_y) < 20:
                                print('click')
                                pyautogui.click()
                                pyautogui.sleep(1)
                            elif abs(index_y - thumb_y) < 100:
                                pyautogui.moveTo(index_x, index_y)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            self.label.config(image=frame)
            self.label.image = frame
            self.root.update()
            
        cv2.destroyAllWindows()
            
    def stop_mouse(self):
        self.root.destroy()

# Exemple d'utilisation :
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    virtual_mouse = VirtualMouse(cap)
    virtual_mouse.start_mouse()
