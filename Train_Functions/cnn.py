import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_data(image_dir, image_size=(224, 224)):
    """Charger les données et préparer les ensembles d'entraînement et de validation"""
    # Charger les fichiers CSV et enlever la première ligne
    X_train = pd.read_csv('X_train.csv', skiprows=1, usecols=[0, 1, 2, 3, 4])
    Y_train = pd.read_csv('Y_train.csv', skiprows=1)

    # Renommer les colonnes pour une manipulation plus facile
    X_train.columns = ['num_produit', 'info1', 'info2', 'info3', 'image_id']
    Y_train.columns = ['num_produit', 'categorie']

    # Supprimer les lignes avec des valeurs manquantes
    X_train.dropna(subset=['image_id'], inplace=True)
    Y_train.dropna(subset=['categorie'], inplace=True)

    # Fusionner les données d'entraînement pour obtenir les labels
    train_data = pd.merge(X_train[['num_produit', 'image_id']], Y_train, on='num_produit')

    # Ajouter le chemin complet des images
    train_data['image_path'] = train_data['image_id'].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))

    # Diviser les données en ensembles d'entraînement et de validation
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Préparer les générateurs de données d'image
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_data,
        x_col='image_path',
        y_col='categorie',
        target_size=image_size,
        batch_size=32,
        class_mode='raw'
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_data,
        x_col='image_path',
        y_col='categorie',
        target_size=image_size,
        batch_size=32,
        class_mode='raw'
    )

    return train_generator, val_generator

def build_model(num_labels):
    """Définir et compiler le modèle CNN pré-entraîné pour la classification d'images"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_labels, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(train_generator, val_generator, num_labels):
    """Entraîne le modèle CNN et évalue ses performances"""
    model = build_model(num_labels)

    # Entraîner le modèle
    history = model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Évaluer le modèle
    val_predictions = model.predict(val_generator)
    val_predictions = np.argmax(val_predictions, axis=1)
    val_labels = val_generator.labels

    print("Accuracy:", accuracy_score(val_labels, val_predictions))
    print(classification_report(val_labels, val_predictions, target_names=[str(i) for i in range(num_labels)]))

    return model, history

def main():
    """Fonction principale pour charger les données, entraîner et évaluer le modèle"""
    image_dir = 'images_data'
    train_generator, val_generator = load_data(image_dir)
    num_labels = len(train_generator.class_indices)
    model, history = train_and_evaluate(train_generator, val_generator, num_labels)

    # Sauvegarder le modèle entraîné
    model.save('cnn_image_classification_model.h5')
    print("Modèle entraîné et sauvegardé avec succès.")

# Appeler la fonction principale
if __name__ == "__main__":
    main()
