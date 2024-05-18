def rf_train(teacher_mode) :
    """Entraine le modèle decision tree et le sauvegarde"""
    import TextProcessing as TP
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from joblib import dump

    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)
    rf_param_grid = {
    'n_estimators': [300, 400, 500, 600], 
    }
    # Configurer et entraîner le modèle Random Forest
    rf_model = RandomForestClassifier()
    rf_random_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='f1_micro', n_jobs=-1, verbose=10)
    rf_random_search.fit(X_train, Y_train)

    # Affichage du meilleur modèle et de ses paramètres
    best_rf_model = rf_random_search.best_estimator_
    print("Meilleurs paramètres de la forêt aléatoire:", rf_random_search.best_params_)
    print("Meilleure précision obtenue:", rf_random_search.best_score_)
    
    # Sauvegarde du modèle
    dump([best_rf_model, X_train, X_test, Y_train, Y_test], 'Trained_Model/rf.model') 

def rf_predict():
    """Effectue une prédiction à partir de rf.model"""
    from joblib import load
    from sklearn.metrics import f1_score
    try :
        rf_model, X_train, X_test, Y_train, Y_test = load('Trained_Model/rf.model') 
    except FileNotFoundError :
        print("entrainez d'abord le modèle avec la fonction rf_train()")
    # Prédiction et évaluation
    Y_pred_rf = rf_model.predict(X_test)
    print(f1_score(Y_test,Y_pred_rf, average="micro"))
    return Y_pred_rf

rf_predict()