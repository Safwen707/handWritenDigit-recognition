import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Charger le dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Préparer les données comme dans le code d'entraînement
X_train = X_train.reshape(X_train.shape[0], 28 * 28) / 255.0
X_test = X_test.reshape(X_test.shape[0], 28 * 28) / 255.0
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Initialiser et entraîner le modèle
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train_one_hot, validation_split=0.3, epochs=10, batch_size=32)

# Effectuer des prédictions sur de nouvelles données
# Exemple : prédire les labels pour le jeu de test
predictions = model.predict(X_test)

# Afficher les prédictions pour les 10 premiers échantillons
for i in range(10):
    predicted_label = np.argmax(predictions[i])  # Trouver la classe avec la plus haute probabilité
    true_label = y_test[i]  # Étiquette réelle
    print(f"Échantillon {i+1}: Prédit = {predicted_label}, Réel = {true_label}")
model.save("mnist_model.h5")