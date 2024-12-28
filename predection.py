from PIL import Image  
import numpy as np  
import matplotlib.pyplot as plt  
from tensorflow.keras.models import load_model  # pour importer le modèle deja sauvgarder


model = load_model("mnist_model.h5")
import time

import pyscreenshot as ImageGrab
n=int(input("Enter the number of images you want to predict: "))
# Définir la région de capture de l'écran (x1, y1, x2, y2)
# x1, y1 : coordonnées du coin supérieur gauche de la région
# x2, y2 : coordonnées du coin inférieur droit de la région
region = (70, 300, 500, 600)
for i in range(n):
    #capture d'ecran
    print("trtacer l'image")
    time.sleep(7)
    screenshot = ImageGrab.grab(bbox=region)
    
    # Pretraitement de l'image
    new_image = screenshot.convert("L")  # Convertir en niveaux de gris (L = luminance)
    new_image = new_image.resize((28, 28))  

    
    new_image_array = np.array(new_image)  
    new_image_array = new_image_array.reshape(1, 784)  # Aplatir pour correspondre au format d'entrée du modèle
    new_image_array = new_image_array / 255.0  # Normaliser les valeurs des pixels entre 0 et 1

    #prediction
    new_prediction = model.predict(new_image_array)  # les probabilités pour chaque classe
    print(new_prediction)
    predicted_label = np.argmax(new_prediction)  # la classe de probabilité plus élevée	

   

    # Afficher l'image et le résultat
    plt.imshow(new_image, cmap="gray")  
    plt.title(f"Classe prédite : {predicted_label}")  
    plt.axis("off")  
    plt.show()
