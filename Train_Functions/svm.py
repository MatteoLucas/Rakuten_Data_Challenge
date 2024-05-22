def svc_train(teacher_mode):
    """Entraine le modèle SVC et le sauvegarde"""
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    from sklearn.svm import SVC
    import TextProcessing as TP

    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)

    # Définition du modèle SVC avec les meilleurs paramètres trouvés
    best_params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
    svc = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'], verbose=10)

    # Entraîner le modèle avec les données d'entraînement
    svc.fit(X_train, Y_train)

    # Sauvegarder le modèle entraîné
    TP.save_model([svc, X_train, X_test, Y_train, Y_test], 'svc', teacher_mode) 

    print("Modèle entraîné et sauvegardé avec succès.")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("argument", type=str)
    args = parser.parse_args()
    svc_train(args.argument.lower() == 'true')