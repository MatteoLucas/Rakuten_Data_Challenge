def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')

    return string

def raw_to_tokens(raw_string, spacy_nlp):
    # Write code for lower-casing
  spacy_tokens = raw_string.lower()

    # Write code to normalize the accents
  string_tokens = normalize_accent(spacy_tokens)

    # Write code to tokenize
  spacy_tokens = spacy_nlp(string_tokens)

    # Write code to remove punctuation tokens and create string tokens
  string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]

    # Write code to join the tokens back into a single string
  clean_string = " ".join(string_tokens)

  return clean_string


def docs_to_tfidf(docs_raw) :
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    spacy_nlp = spacy.load("fr_core_news_sm")
    docs_clean = [raw_to_tokens(doc, spacy_nlp) for doc in docs_raw]
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(docs_clean)
    return X_tfidf, tfidf

def ouverture_fichier(file_name):
    import pandas as pd
    # Lire les données depuis le fichier CSV
    data = pd.read_csv(file_name, header=0, sep=',', decimal='.')
    # Supprimer les colonnes contenant uniquement des valeurs manquantes
    data = data.dropna(axis=1, how='all')
    # Supprimer les lignes contenant uniquement des valeurs manquantes
    data = data.dropna(axis=0, how='all')
    # Afficher des informations sur le DataFrame, telles que le nombre de lignes et de colonnes,
    # ainsi que le type de données de chaque colonne
    print(data.info())
    # Afficher les dimensions du jeu de données après le nettoyage
    print('Dimensions du jeu de données :', data.shape)
    return data

def create_X_train_tfidf() :
    import numpy as np
    import pickle
    from scipy import sparse
    X_train = ouverture_fichier("Data/X_train.csv")
    # Supprimer les colonnes innutiles
    X_train = X_train.drop(columns=['description', 'productid', 'imageid'])
    X_train_tfidf, tfidf= docs_to_tfidf(X_train['designation'][:100])
    print("Shape of the TF-IDF Matrix:")
    print(X_train_tfidf.shape)
    with open('Matrix&Model/X_train_tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    sparse.save_npz('Matrix&Model/X_train_tfidf_matrix.npz', X_train_tfidf)