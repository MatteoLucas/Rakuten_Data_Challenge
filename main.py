def majority_vote(models, teacher_mode):
    """
    Combine les prédictions de plusieurs modèles par vote majoritaire
    Args :
        model (list): Liste des modèles votant
        teacher_mode (bool): is the teacher testing ?
    """
    from scipy.stats import mode
    import numpy as np
    from sklearn.metrics import f1_score
    import TextProcessing as TP

    # Empiler les prédictions en une seule matrice
    Y_pred, Y_test, X_train = predict(models[0], teacher_mode)
    Y_preds = np.vstack([Y_pred]+[predict(model, teacher_mode)[0] for model in models[1:]])
    
    # Calculer le vote majoritaire pour chaque échantillon
    Y_final_pred_full, counts = mode(Y_preds, axis=0)
    
    # En cas d'égalité, retenir la prédiction du premier modèle
    ties = (counts == 1)
    Y_final_pred_full[ties] = Y_preds[0, ties]

    Y_final_pred = Y_final_pred_full.ravel()

    model_name = ""
    for model in models:
        model_name+=model+"+"
    model_name = model[:-1]

    if not teacher_mode :
        print("F1 score macro : ",f1_score(Y_test,Y_final_pred, average="macro"))
    else :
        TP.save_predictions_to_csv(Y_final_pred, "Y_pred_"+model_name+".csv", X_train)

    return Y_final_pred

majority_vote(['svc','rn','knn'], False)