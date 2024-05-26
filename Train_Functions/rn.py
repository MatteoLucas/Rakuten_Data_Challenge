def rn_train(teacher_mode = True, svd=False):
    """Entraine un réseau de neurones pour une classification multi-classes et le sauvegarde"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP

    if teacher_mode :
        print("Ce modèle ne peut-etre entrainé qu'en teacher_mode=False car il a besoin de Y_test pour la loss function.\nCependant, une prédiction sera possible sur tout le jeu de donné")
        teacher_mode = False
        print("Entrainement en teacher_mode=False")

    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode, svd)

    # Assurez-vous que Y_train et Y_test sont one-hot encoded
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # Définir le nombre de classes
    num_classes = Y_train.shape[1]  # Assurez-vous que Y_train est one-hot encoded

    # Définir la structure du réseau de neurones avec hyperparamètres ajustables
    model = Sequential()
    
    # Ajout de la première couche dense
    model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.3))
 
    # Ajout d'une deuxième couche dense
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    # Ajout d'une troisième couche dense
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    # Couche de sortie
    model.add(Dense(num_classes, activation='softmax'))  # Utiliser 'softmax' pour une classification multi-classes

    # Compiler le modèle avec des hyperparamètres ajustables
    model.compile(loss='categorical_crossentropy',  # Utiliser 'categorical_crossentropy' pour une classification multi-classes
                  optimizer=Adam(learning_rate=0.0005),
                  metrics=['f1_score'])

    # Entraîner le modèle avec des hyperparamètres ajustables
    model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))

    # Sauvegarder le modèle entraîné
    if not svd :
        TP.save_model([model, X_train, X_test, Y_train, Y_test], 'rn', teacher_mode)
    else :
        TP.save_model([model, X_train, X_test, Y_train, Y_test], 'svd_rn', teacher_mode)

    print("Modèle entraîné et sauvegardé avec succès.")



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_mode", type=str)
    parser.add_argument("--svd", type=str, default="False")
    args = parser.parse_args()
    if args.teacher_mode == None : teacher_mode = "True"
    rn_train(teacher_mode=teacher_mode.lower() == 'true',svd= args.svd.lower() == 'true')