def knn_train(teacher_mode=True, svd=False):
    """Entraine le modèle KNN avec validation croisée et le sauvegarde"""
    from sklearn.neighbors import KNeighborsClassifier
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    
    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(teacher_mode, svd)
    
    # Définition des paramètres pour la validation croisée
    best_param = {'n_neighbors': 35,'weights': 'uniform','metric':'euclidean'} #Trouvés par validation croisée
    
    # Initialisation du modèle KNN
    knn = KNeighborsClassifier(n_neighbors=best_param['n_neighbors'], weights=best_param['weights'], metric=best_param['metric'])
        
    # Entraîner le modèle avec les données d'entraînement
    knn.fit(X_train, Y_train)
    
    # Sauvegarder le modèle entraîné
    if not svd :
        TP.save_model([knn, X_train, X_test, Y_train, Y_test], 'knn', teacher_mode)
    else :
        TP.save_model([knn, X_train, X_test, Y_train, Y_test], 'svd_knn', teacher_mode)
    
    print("Modèle entraîné et sauvegardé avec succès.")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_mode", type=str)
    parser.add_argument("--svd", type=str, default="False")
    args = parser.parse_args()
    if args.teacher_mode == None : teacher_mode = "True"
    knn_train(teacher_mode=teacher_mode.lower() == 'true',svd= args.svd.lower() == 'true')