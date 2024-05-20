def svc_train(teacher_mode):
    """Entraine le modèle SVC et le sauvegarde"""
    from joblib import dump
    from sklearn.svm import SVC
    import TextProcessing as TP

    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)

    # Définition du modèle SVC avec les meilleurs paramètres trouvés
    best_params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
    svc = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])

    # Entraîner le modèle avec les données d'entraînement
    svc.fit(X_train, Y_train)

    # Sauvegarder le modèle entraîné
    TP.save_model([svc, X_train, X_test, Y_train, Y_test], 'svc') 

    print("Modèle entraîné et sauvegardé avec succès.")