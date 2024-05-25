def find_divisors(n):
    divisors = []
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors

def closest_divisor(x, y):
    divisors = find_divisors(x)
    closest = divisors[0]
    min_diff = abs(divisors[0] - y)
    
    for divisor in divisors:
        diff = abs(divisor - y)
        if diff < min_diff:
            min_diff = diff
            closest = divisor
    
    return closest

def svd_train(n_components = 30000):
    """Entraine le modèle réduction de dimension et le sauvegarde"""
    import sys
    import os
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP

    # Récupérer les données d'entraînement et de test
    X_train, X_test, Y_train, Y_test = TP.get_X_Y(True)

    # Convertir les données en float32 pour économiser de la mémoire
    X = X_train.astype(np.float32)


    # Définir le nombre de composants pour TruncatedSVD
    svd = TruncatedSVD(n_components=n_components)
    x_size = X.shape[0]

    # Diviser les données d'entraînement en batches
    batch_size = closest_divisor(x_size, 1000)
    print("Batch size : ",batch_size)

    for i in range(0, X.shape[0], batch_size):
        print("svd fiting : ",i,"/",x_size ) 
        svd.fit_transform(X[i:i + batch_size])

    TP.save_model([svd, None, None, None, None], 'svd', False) 

def svd_fit(X):
    import sys
    import os
    # Ajouter le chemin du dossier parent
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(parent_dir)
    import TextProcessing as TP
    import numpy as np

    svd, X_train, X_test, Y_train, Y_test = TP.load_model('svd', False) 
    # Diviser les données d'entraînement en batches
    x_size = X.shape[0]
    batch_size = closest_divisor(x_size, 1000)
    print("Batch size : ",batch_size)

    # Diviser les données de test en batches et transformer
    X_transformed = []
    for i in range(0, X.shape[0], batch_size):
        print("svd transforming : ",i,"/",x_size )
        X_batch = X[i:i + batch_size]
        X_batch_transformed = svd.transform(X_batch)
        X_transformed.append(X_batch_transformed)

    return np.vstack(X_transformed)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=30000)
    args = parser.parse_args()
    svd_train(args.n_components)
