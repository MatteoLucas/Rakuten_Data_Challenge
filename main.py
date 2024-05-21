def predict(model_name, teacher_mode):
    '''
    Fonction pour faire une prédiction à partir des modèles entrainés
    Args:
        model_name (str): Name of the model (rf, svc, dt, )
        teacher_mode (bool): is the teacher testing ?
    '''
    from sklearn.metrics import f1_score
    import TextProcessing as TP
    import numpy as np

    model, X_train, X_test, Y_train, Y_test = TP.load_model(model_name, teacher_mode) 

    if teacher_mode :
        X_train, X_test, Y_train, Y_test = TP.get_X_Y(True) #Pour etre sur que la prédiction sera sur le bon X_test

    # Prédiction et évaluation
    Y_pred = model.predict(X_test)

    # Convertir Y_pred en labels si nécessaire
    if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
        Y_pred = np.argmax(Y_pred, axis=1)
    # Convertir Y_test en labels si nécessaire
    if not teacher_mode and Y_test.ndim > 1 and Y_test.shape[1] > 1:
        Y_test = np.argmax(Y_test, axis=1)

    if not teacher_mode :
        print("F1 score macro : ",f1_score(Y_test,Y_pred, average="macro"))
    else :
        TP.save_predictions_to_csv(Y_pred, "Y_pred_"+model_name+".csv", X_train)
    return Y_pred

predict('rn', True)