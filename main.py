def predict(model_name, teacher_mode):
    '''
    Fonction pour faire une prédiction à partir des modèles entrainés
    Args:
        model_name (str): Name of the model (rf, svc, dt, )
        teacher_mode (bool): is the teacher testing ?
    '''
    from sklearn.metrics import f1_score
    import TextProcessing as TP

    model, X_train, X_test, Y_train, Y_test = TP.load_model(model_name, teacher_mode) 

    if teacher_mode :
        X_train, X_test, Y_train, Y_test = TP.get_X_Y(True) #Pour etre sur que la prédiction sera sur le bon X_test

    # Prédiction et évaluation
    Y_pred = model.predict(X_test)
    if not teacher_mode :
        print("F1 score macro : ",f1_score(Y_test,Y_pred, average="micro"))
    TP.save_predictions_to_csv(Y_pred, "Y_pred_"+model_name+".csv")
    return Y_pred

predict('rf', False)