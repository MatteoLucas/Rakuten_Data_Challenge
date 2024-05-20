def rf_train(teacher_mode) :
    """Entraine le modèle Random Forest et le sauvegarde"""
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