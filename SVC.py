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

def svc_predict(teacher_mode):
    """Effectue une prédiction à partir de rf.model"""
    from sklearn.metrics import f1_score
    import sys
    import TextProcessing as TP
    try :
        svc_model, X_train, X_test, Y_train, Y_test = TP.load_model('svc') 
    except FileNotFoundError :
        print("Entrainez d'abord le modèle avec la fonction rf_train()")
        sys.exit()

    if teacher_mode :
        X_train, X_test, Y_train, Y_test = TP.get_X_Y(True) #Pour etre sur que la prédiction sera sur le bon X_test

    # Prédiction et évaluation
    Y_pred = svc_model.predict(X_test)
    if not teacher_mode :
        print(f1_score(Y_test,Y_pred, average="micro"))
    TP.save_predictions_to_csv(Y_pred, "Y_pred_svc.csv")
    return Y_pred

if __name__ == "__main__":
    svc_predict(False)