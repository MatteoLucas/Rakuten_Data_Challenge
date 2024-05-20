def rf_train(teacher_mode) :
    """Entraine le modèle Random Forest et le sauvegarde"""
    import TextProcessing as TP
    from sklearn.ensemble import RandomForestClassifier
    from joblib import dump

    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)

    # Définition du modèle Random Forest avec les meilleurs paramètres trouvés
    best_param = {'n_estimators': 800}
    rf_model = RandomForestClassifier(n_estimators=best_param['n_estimators'])
    rf_model.fit(X_train, Y_train)
    
    # Sauvegarde du modèle
    dump([rf_model, X_train, X_test, Y_train, Y_test], 'Trained_Model/rf.model') 
    print("Modèle entraîné et sauvegardé avec succès.")

def rf_predict(teacher_mode):
    """Effectue une prédiction à partir de rf.model"""
    from joblib import load
    from sklearn.metrics import f1_score
    import sys
    import TextProcessing as TP
    try :
        rf_model, X_train, X_test, Y_train, Y_test = load('Trained_Model/rf.model') 
    except FileNotFoundError :
        print("Entrainez d'abord le modèle avec la fonction rf_train()")
        sys.exit()

    # Prédiction et évaluation
    Y_pred_rf = rf_model.predict(X_test)
    if not teacher_mode :
        print(f1_score(Y_test,Y_pred_rf, average="micro"))
    TP.save_predictions_to_csv(Y_pred_rf, "Y_pred_rf.csv")
    return Y_pred_rf

rf_predict()