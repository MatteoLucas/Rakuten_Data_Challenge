
import numpy as np
import sys
import pickle
from scipy import sparse

try : 
    with open('Matrix&Model/X_train_tfidf_model.pkl', 'rb') as f:
        X_train_loaded = pickle.load(f)
    X_train_tfidf_matrix = sparse.load_npz('Matrix&Model/X_train_tfidf_matrix.npz')
except FileNotFoundError :
    print("Reportez vous au fichier README.md pour générer la matrice X_train_tfidf")
    sys.exit()

print("TF-IDF Matrix:")
print(X_train_tfidf_matrix.todense())