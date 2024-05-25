def gb_train(teacher_mode, svd=False) :
    """Entraine le modèle Gradient Boosting et le sauvegarde"""
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    from sklearn.ensemble import GradientBoostingClassifier
    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode, svd)

    # Définition du modèle gradient boosting avec les meilleurs paramètres trouvés
    best_param = {'learning_rate': 0.1,'max_depth': 10,'n_estimators': 250}
    gb = GradientBoostingClassifier(learning_rate=best_param['learning_rate'],max_depth=best_param['max_depth'],n_estimators=best_param['n_estimators'], verbose=10)
    
    # Entraîner le modèle avec les données d'entraînement
    gb.fit(X_train, Y_train)

    # Sauvegarder le modèle entraîné
    if not svd :
        TP.save_model([gb, X_train, X_test, Y_train, Y_test], 'gb', teacher_mode)
    else :
        TP.save_model([gb, X_train, X_test, Y_train, Y_test], 'gp_svm', teacher_mode)

    print("Modèle entraîné et sauvegardé avec succès.")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_mode", type=str)
    parser.add_argument("--svd", type=str, default="False")
    args = parser.parse_args()
    gb_train(args.teacher_mode.lower() == 'true', args.svd.lower() == 'true')