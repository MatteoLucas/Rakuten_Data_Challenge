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
    
    if model_name == "svd":
        raise TypeError("svd n'est pas un modèle pour faire des prédictions, il sert à faire une réduction de la dimension lors de l'entrainement d'autres modèles")


    model, X_train, X_test, Y_train, Y_test = TP.load_model(model_name, teacher_mode) 

    if teacher_mode :
        if len(model_name)>3 and model_name[:3] == "svd":
            svd = True
        else :
            svd = False
        X_train, X_test, Y_train, Y_test = TP.get_X_Y(True, svd) #Pour etre sur que la prédiction sera sur le bon X_test
    
    print("Predicting ...")
    
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
    return Y_pred, Y_test, X_train

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("teacher_mode", type=str)
    args = parser.parse_args()
    predict(args.model,args.teacher_mode.lower() == 'true')