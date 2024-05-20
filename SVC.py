def svc_train() :
    """Entraine le modèle svc et le sauvegarde"""
    from joblib import dump
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    import TextProcessing as TP
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(False)

    # Définir le modèle SVC
    svc = SVC()

    # Définir la grille des hyperparamètres à rechercher
    param_grid = {
        'C': [200, 100, 300],
        'gamma': [1, 5, 10],
        'kernel': ['rbf']
    }

    # Mettre en place la recherche par validation croisée
    grid_search = GridSearchCV(svc, param_grid, refit=True, verbose=10, cv=3, n_jobs=-1)

    # Entraîner le modèle avec les données d'entraînement
    grid_search.fit(X_train, Y_train)

    # Afficher les meilleurs paramètres trouvés par la validation croisée
    print(f'Best Parameters: {grid_search.best_params_}')

    # Prédire avec le meilleur modèle trouvé
    best_model = grid_search.best_estimator_
    # Sauvegarde du modèle
    dump([best_model, X_train, X_test, Y_train, Y_test], 'Trained_Model/svc.model')

def svc_predict():
    """Effectue une prédiction à partir de rf.model"""
    from joblib import load
    from sklearn.metrics import f1_score
    try :
        svc_model, X_train, X_test, Y_train, Y_test = load('Trained_Model/svc.model') 
    except FileNotFoundError :
        print("entrainez d'abord le modèle avec la fonction rf_train()")
    # Prédiction et évaluation
    Y_pred = svc_model.predict(X_test)

    print(f1_score(Y_test,Y_pred, average="micro"))
    return Y_pred