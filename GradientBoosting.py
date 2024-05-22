def gb_train(teacher_mode) :
    """Entraine le modèle decision tree et le sauvegarde"""
    import TextProcessing as TP
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from joblib import dump

    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)
    gb_param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'max_depth': [3, 6, 10],
    'n_estimators': [250, 400],
    }
    # Configurer et entraîner le modèle Random Forest
    gb_model = GradientBoostingClassifier()
    gb_random_search = GridSearchCV(estimator=gb_model, param_grid=gb_param_grid, cv=2, scoring='f1_micro', n_jobs=-1, verbose=10)
    gb_random_search.fit(X_train, Y_train)

    # Affichage du meilleur modèle et de ses paramètres
    best_gb_model = gb_random_search.best_estimator_
    print("Meilleurs paramètres du gradient boosting :", gb_random_search.best_params_)
    print("Meilleure précision obtenue:", gb_random_search.best_score_)
    
    # Sauvegarde du modèle
    dump([best_gb_model, X_train, X_test, Y_train, Y_test], 'Trained_Model/gb.model') 

def gb_predict():
    """Effectue une prédiction à partir de gb.model"""
    from joblib import load
    from sklearn.metrics import f1_score
    import sys
    try :
        gb_model, X_train, X_test, Y_train, Y_test = load('Trained_Model/gb.model') 
    except FileNotFoundError :
        print("Entrainez d'abord le modèle avec la fonction gb_train()")
        sys.exit()
    # Prédiction et évaluation
    Y_pred_gb = gb_model.predict(X_test)
    print(f1_score(Y_test,Y_pred_gb, average="micro"))
    return Y_pred_gb


if __name__=="__main__" :
    gb_train(False)