import os
import cv2
from pynput import keyboard

SAVE_DIR = "../captures/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_next_sequence_number(key):
    files = [f for f in os.listdir(SAVE_DIR) if f.startswith(key)]
    if not files:
        return 1 
    
    numbers = [int(f.split('(')[1].split(')')[0]) for f in files]
    return max(numbers) + 1

def capture_webcam_photo(frame, key):
    sequence_number = get_next_sequence_number(key)
    filename = f"{key}({sequence_number}).png"
    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"Foto guardada: {filepath}")

def on_press(key):
    try:
        if key.char.isalpha():
            capture_webcam_photo(last_frame, key.char.lower()) 
    except AttributeError:
        pass

cap = cv2.VideoCapture(0)


listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Webcam Preview - pulsa la tecla para guardar", frame)
    last_frame = frame
    #para salir le presionas al "."
    if cv2.waitKey(1) & 0xFF == ord('.'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
