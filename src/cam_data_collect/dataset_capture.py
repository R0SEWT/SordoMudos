import os
import cv2
from pynput import keyboard

SAVE_DIR = "../../data/captures/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_next_sequence_number(key_dir, key):
    # Listar archivos que empiezan con la letra de la tecla dentro del directorio
    files = [f for f in os.listdir(key_dir) if f.startswith(key)]
    if not files:
        return 1
    
    # Obtener el número más alto de la secuencia
    numbers = [int(f.split('(')[1].split(')')[0]) for f in files]
    return max(numbers) + 1

def capture_webcam_photo(frame, key):
    key_dir = os.path.join(SAVE_DIR, key)
    
    if not os.path.exists(key_dir):
        os.makedirs(key_dir)
    
    sequence_number = get_next_sequence_number(key_dir, key)
    filename = f"{key}({sequence_number}).png"
    filepath = os.path.join(key_dir, filename)
    
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
    
    if cv2.waitKey(1) & 0xFF == ord('.'):
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
