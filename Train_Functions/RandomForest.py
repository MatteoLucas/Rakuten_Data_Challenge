def rf_train(teacher_mode) :
    """Entraine le modèle Random Forest et le sauvegarde"""
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)

    # Définition du modèle Random Forest avec les meilleurs paramètres trouvés
    best_param = {'n_estimators': 800}
    rf_model = RandomForestClassifier(n_estimators=best_param['n_estimators'])
    rf_model.fit(X_train, Y_train)
    
    # Sauvegarde du modèle
    TP.save_model([rf_model, X_train, X_test, Y_train, Y_test], 'rf', teacher_mode) 
    print("Modèle entraîné et sauvegardé avec succès.")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("argument", type=bool)
    args = parser.parse_args()
    rf_train(args.argument)