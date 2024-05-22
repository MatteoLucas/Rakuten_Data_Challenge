def gb_train(teacher_mode) :
    """Entraine le modèle Gradient Boosting et le sauvegarde"""
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    from sklearn.ensemble import GradientBoostingClassifier
    from joblib import dump
    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)

    # Définition du modèle gradient boosting avec les meilleurs paramètres trouvés
    best_param = {'learning_rate': 0.1,'max_depth': 6,'n_estimators': 400}
    gb = GradientBoostingClassifier(learning_rate=best_param['learning_rate'],max_depth=best_param['max_depth'],n_estimators=best_param['n_estimators'], n_jobs=-1, verbose=10)
    
    # Entraîner le modèle avec les données d'entraînement
    gb.fit(X_train, Y_train)

    # Sauvegarder le modèle entraîné
    TP.save_model([gb, X_train, X_test, Y_train, Y_test], 'gb', teacher_mode) 

    print("Modèle entraîné et sauvegardé avec succès.")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("argument", type=str)
    args = parser.parse_args()
    gb_train(args.argument.lower() == 'true')