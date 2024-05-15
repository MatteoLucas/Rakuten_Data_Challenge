def dt_train_and_predict(teacher_mode) :
    """Renvoie la prediction et le model pour un arbre de décision si on est en teacher mode"""
    import TextProcessing as TP
    # Pour la visualisation des données
    import matplotlib.pyplot as plt  # Pour réaliser des graphiques
    from sklearn.model_selection import GridSearchCV
    # Pour les arbres de décision
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import classification_report

    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode)
    # Création du modèle de l'arbre de décision avec GridSearchCV
    dt_model = DecisionTreeClassifier()
    #dt_grid_search = GridSearchCV(estimator=dt_model, param_grid=dt_param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    dt_model.fit(X_train, Y_train)

    # Affichage du meilleur modèle et de ses paramètres
    best_dt_model = dt_model
    #print("Meilleurs paramètres de l'arbre de décision:", dt_grid_search.best_params_)
    #print("Meilleure précision obtenue:", dt_grid_search.best_score_)

    # Visualisation de l'arbre (pour les arbres pas trop grands)
    if dt_model.tree_.node_count < 100:  # Ajuster ce seuil si nécessaire
        plt.figure(figsize=(20,10))
        plot_tree(dt_model, filled=True)
        plt.show()

    # Prédiction et évaluation
    Y_pred_dt = dt_model.predict(X_test)

    if not teacher_mode:
        print(classification_report(Y_test, Y_pred_dt))

    Y_pred_dt, dt_model
