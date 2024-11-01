import cv2
import numpy as np
import os

def to_silhouette(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sse le aplica el umbral para segmentar la mano
    _, silhouette = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(silhouette) > 127:
        silhouette = cv2.bitwise_not(silhouette)
    return silhouette

def binarize_with_canny(img, weak_th = None, strong_th = None): 
	
	if img is None:
		raise ValueError("La imagen no puede ser None")
	# conversion of image to grayscale 
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	
	# Noise reduction step 
	img = cv2.GaussianBlur(img, (5, 5), 1.4) 
	
	# Calculating the gradients 
	gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
	gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3) 
	
	# Conversion of Cartesian coordinates to polar 
	mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True) 
	
	# setting the minimum and maximum thresholds 
	# for double thresholding 
	mag_max = np.max(mag) 
	if not weak_th:weak_th = mag_max * 0.1
	if not strong_th:strong_th = mag_max * 0.5
	
	# getting the dimensions of the input image 
	height, width = img.shape 
	
	# Looping through every pixel of the grayscale 
	# image 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_ang = ang[i_y, i_x] 
			grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang) 
			
			# selecting the neighbours of the target pixel 
			# according to the gradient direction 
			# In the x axis direction 
			if grad_ang<= 22.5: 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# top right (diagonal-1) direction 
			elif grad_ang>22.5 and grad_ang<=(22.5 + 45): 
				neighb_1_x, neighb_1_y = i_x-1, i_y-1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
			
			# In y-axis direction 
			elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90): 
				neighb_1_x, neighb_1_y = i_x, i_y-1
				neighb_2_x, neighb_2_y = i_x, i_y + 1
			
			# top left (diagonal-2) direction 
			elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135): 
				neighb_1_x, neighb_1_y = i_x-1, i_y + 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y-1
			
			# Now it restarts the cycle 
			elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180): 
				neighb_1_x, neighb_1_y = i_x-1, i_y 
				neighb_2_x, neighb_2_y = i_x + 1, i_y 
			
			# Non-maximum suppression step 
			if width>neighb_1_x>= 0 and height>neighb_1_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]: 
					mag[i_y, i_x]= 0
					continue

			if width>neighb_2_x>= 0 and height>neighb_2_y>= 0: 
				if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]: 
					mag[i_y, i_x]= 0

	weak_ids = np.zeros_like(img) 
	strong_ids = np.zeros_like(img)			 
	ids = np.zeros_like(img) 
	
	# double thresholding step 
	for i_x in range(width): 
		for i_y in range(height): 
			
			grad_mag = mag[i_y, i_x] 
			
			if grad_mag<weak_th: 
				mag[i_y, i_x]= 0
			elif strong_th>grad_mag>= weak_th: 
				ids[i_y, i_x]= 1
			else: 
				ids[i_y, i_x]= 2
	
	
	return mag 


def save_silhouette(input_dir=os.path.join(os.path.dirname(__file__), '..', 'augmented_images'), output_dir=os.path.join(os.path.dirname(__file__), '..', 'images_processed'), for_canny=False):
    images = []
    labels = []
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Recorrer todas las imágenes del directorio de entrada
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if for_canny:
                    silhouette = binarize_with_canny(img)
                else:
                    silhouette = to_silhouette(img)

                # Extraer la etiqueta desde el nombre de la carpeta
                label = os.path.basename(os.path.dirname(img_path))
                labels.append(label)

                # Obtener el nombre original del archivo y agregar "_s" antes de la extensión
                name, ext = os.path.splitext(file)
                new_name = f"{name}_s{ext}"
                # Crear la estructura de carpetas en la salida
                relative_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # Guardar la imagen de silueta en la carpeta correspondiente
                save_path = os.path.join(output_path, new_name)
                cv2.imwrite(save_path, silhouette)
                
                # Agregar la imagen a la lista
                images.append(silhouette)
    
    return images, labels


im = cv2.imread('images/a_original.jpg')
can = binarize_with_canny(im)

