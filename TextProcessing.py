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


def docs_to_tfidf(docs_raw, tfidf) :
  import spacy
  from sklearn.feature_extraction.text import TfidfVectorizer
  spacy_nlp = spacy.load("fr_core_news_sm")
  docs_clean = [raw_to_tokens(doc, spacy_nlp) for doc in docs_raw]
  if tfidf != None :
    tfidf = TfidfVectorizer()
  X_tfidf = tfidf.fit_transform(docs_clean)
  return X_tfidf, tfidf

def ouverture_fichier(file_name):
  import pandas as pd
  # Lire les données depuis le fichier CSV
  data = pd.read_csv(file_name, header=0, sep=',', decimal='.')
  return data

def create_X_train_tfidf(file, tfidf=None) :
  import pickle
  from scipy import sparse
  #On ouvre le fichier
  X_train = ouverture_fichier(file)
  # Supprimer les colonnes innutiles
  X_train = X_train.drop(columns=['description', 'productid', 'imageid'])

  X_train_tfidf, tfidf= docs_to_tfidf(X_train['designation'], tfidf = tfidf)
  print("Shape of the TF-IDF Matrix:")
  print(X_train_tfidf.shape)
  with open('Matrix&Model/X_train_tfidf_model.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
  sparse.save_npz('Matrix&Model/X_train_tfidf_matrix.npz', X_train_tfidf)
  return X_train_tfidf, tfidf


def get_X_Y(teacher_mode) :
  """
  Cette fonction renvoie X_train, X_test, Y_train, Y_test

  Si on est en teacher_mode = False :
  X_test et Y_test sont pris dans X_train.csv et dans Y_train.csv

  Si on est en teacher_mode = True :
  X_test est pris dans X_test.csv et Y_test est None
  """
  import TextProcessing as TP
  import pickle
  from scipy import sparse
  from sklearn.model_selection import train_test_split

  #On essaye d'ouvrir la matrice tfidf de X_train si elle existe sinon, on la crée
  try : 
      with open('Matrix&Model/X_train_tfidf_model.pkl', 'rb') as f:
          X_tfidf_model = pickle.load(f)
      X_tfidf_matrix = sparse.load_npz('Matrix&Model/X_train_tfidf_matrix.npz')
  except FileNotFoundError :
      X_tfidf_matrix, X_tfidf_model = TP.create_X_train_tfidf("Data/X_train.csv")

  #On ouvre Y_train
  Y = TP.ouverture_fichier("Data/Y_train.csv")["prdtypecode"]

  if not teacher_mode :
    '''Si on n'est pas en teacher mode, on sépare X_train et Y_train en un ensemble de train et un ensemble de test'''
    # Définir la proportion de l'ensemble de test
    test_portion = 1/5
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=test_portion, shuffle=True)
    return X_train, X_test, Y_train, Y_test
  
  else :
    '''Si on est en teacher mode on génère la matrice X_test en utilisant le même modèle que pour la X_train'''
    X_test_matrix, X_test_model = TP.create_X_train_tfidf("Data/X_test.csv", tfidf=X_tfidf_model)
    return X_tfidf_matrix, X_test_matrix, Y, None
