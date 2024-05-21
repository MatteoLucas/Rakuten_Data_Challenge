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
  spacy_tokens = str(raw_string).lower()

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
  '''
  Prends comme argument une liste de texte et renvoie la matrice TF-IDF associée
  '''
  import spacy
  from sklearn.feature_extraction.text import TfidfVectorizer
  from joblib import dump
  spacy_nlp = spacy.load("fr_core_news_sm")
  docs_clean = [raw_to_tokens(doc, spacy_nlp) for doc in docs_raw]
  tfidf = TfidfVectorizer()
  X_tfidf = tfidf.fit_transform(docs_clean)
  dump(tfidf, "Matrix&Model/tfidf.vector")
  return X_tfidf


def ouverture_fichier(file_name):
  import pandas as pd
  # Lire les données depuis le fichier CSV
  data = pd.read_csv(file_name, header=0, sep=',', decimal='.')
  return data


def create_X_tfidf() :
  from joblib import dump
  import pandas as pd
  #On ouvre le fichier
  X_train = ouverture_fichier("Data/X_train.csv")
  X_test = ouverture_fichier("Data/X_test.csv")
  longueur_train = len(X_train)
  X = pd.concat([X_train['designation'], X_test['designation']], axis=0)
  X_tfidf= docs_to_tfidf(X)
  X_train_tfidf = X_tfidf[:longueur_train]
  X_test_tfidf = X_tfidf[longueur_train:]
  print("Shape of the TF-IDF Matrix:")
  print("- X_train : ",X_train_tfidf.shape)
  print("- X_test : ",X_test_tfidf.shape)
  dump(X_train_tfidf, 'Matrix&Model/X_train_tfidf.matrix')
  dump(X_test_tfidf, 'Matrix&Model/X_test_tfidf.matrix')
  return X_train_tfidf, X_test_tfidf


def get_X_Y(teacher_mode) :
  """
  Cette fonction renvoie X_train, X_test, Y_train, Y_test

  Si on est en teacher_mode = False :
  X_test et Y_test sont pris dans X_train.csv et dans Y_train.csv

  Si on est en teacher_mode = True :
  X_test est pris dans X_test.csv et Y_test est None
  """
  import TextProcessing as TP
  from joblib import load
  from sklearn.model_selection import train_test_split

  #On essaye d'ouvrir la matrice tfidf de X_train si elle existe sinon, on la crée
  try : 
    X_train_tfidf  = load('Matrix&Model/X_train_tfidf.matrix')
    X_test_tfidf = load('Matrix&Model/X_test_tfidf.matrix')
  except FileNotFoundError :
    X_train_tfidf, X_test_tfidf = TP.create_X_tfidf()

  #On ouvre Y_train
  Y_train = TP.ouverture_fichier("Data/Y_train.csv")["prdtypecode"]

  if not teacher_mode :
    '''Si on n'est pas en teacher mode, on sépare X_train et Y_train en un ensemble de train et un ensemble de test'''
    # Définir la proportion de l'ensemble de test
    test_portion = 1/5
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_tfidf, Y_train, test_size=test_portion, shuffle=True)
    return X_train, X_test, Y_train, Y_test
  
  else :
    '''Si on est en teacher mode on génère la matrice X_test en utilisant le même modèle que pour la X_train'''
    return X_train_tfidf, X_test_tfidf, Y_train, None


def save_predictions_to_csv(Y_pred, csv_name, X_train):
    """Sauvegarde les prédictions dans un fichier CSV."""
    import pandas as pd
    
    # Créer un DataFrame avec les vraies valeurs et les prédictions
    results_df = pd.DataFrame({
        '': [i for i in range(X_train.shape[0],X_train.shape[0]+len(Y_pred))],
        'prdtypecode': Y_pred
    })
    
    # Sauvegarder le DataFrame dans un fichier CSV
    results_df.to_csv("Predictions_ForTeacher/"+csv_name, index=False)
    
    print("Prédictions sauvegardées dans Predictions/"+csv_name)


def split_file(file_base, parts_directory = "Trained_Model/", file_extension = ".model", chunk_size=12 * 1024 * 1024):
  """Splits a binary file into multiple chunks of a given size.
  
  Args:
    parts_directory (str): Directory containing the chunk files.
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
    chunk_size (float): Maximum size of each chunk in bytes (default is 12.5 MB).
  """
  import os

  part_number = 0
  while True:
    chunk_file_name = os.path.join(parts_directory, f"{file_base}_part{part_number}{file_extension}")
    if not os.path.exists(chunk_file_name):
      break
    os.remove(parts_directory+file_base+"_part"+str(part_number)+file_extension)
    part_number += 1  

  file_path = parts_directory+file_base+file_extension
  file_base, file_extension = os.path.splitext(file_path)
  with open(file_path, 'rb') as file:
    chunk_count = 0
    while True:
      chunk = file.read(int(chunk_size))
      if not chunk:
        break
      chunk_file_name = f"{file_base}_part{chunk_count}{file_extension}"
      with open(chunk_file_name, 'wb') as chunk_file:
        chunk_file.write(chunk)
      chunk_count += 1


def merge_files(file_base, parts_directory = "Trained_Model/", file_extension = ".model"):
  """Merges multiple chunk files into a single binary file.
  
  Args:
    parts_directory (str): Directory containing the chunk files.
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
  """
  import os
  output_file = parts_directory+file_base+file_extension
  with open(output_file, 'wb') as merged_file:
    part_number = 0
    while True:
      chunk_file_name = os.path.join(parts_directory, f"{file_base}_part{part_number}{file_extension}")
      if not os.path.exists(chunk_file_name):
        break
      with open(chunk_file_name, 'rb') as chunk_file:
        merged_file.write(chunk_file.read())
      part_number += 1


def load_model(file_base, teacher_mode,  parts_directory = "Trained_Model/", file_extension = ".model") :
  '''
  Args:
    file_base (str): Base name of the chunk files without the part number and extension.
    teacher_mode (bool): is teacher testing ?
    parts_directory (str): Directory containing the chunk files.
    file_extension (str): Extension of the chunk files.
  '''
  from joblib import load
  import os
  if teacher_mode and file_base != "rn":
    parts_directory = "Trained_Model_ForTeacher/"
  if not os.path.exists(parts_directory+file_base+"_part0"+file_extension) :
    raise FileNotFoundError("Entrainez d'abord le modèle avec la fonction "+file_base+"_train()")
  merge_files(file_base, parts_directory, file_extension)
  model, X_train, X_test, Y_train, Y_test = load(parts_directory+file_base+file_extension)
  os.remove(parts_directory+file_base+file_extension)
  return model, X_train, X_test, Y_train, Y_test


def save_model(model_list, file_base, teacher_mode, parts_directory = "Trained_Model/", file_extension = ".model") :
  '''
  Args:
    model_list (list): [svc, X_train, X_test, Y_train, Y_test]
    parts_directory (str): Directory containing the chunk files.
    teacher_mode (bool): is teacher testing ?
    file_base (str): Base name of the chunk files without the part number and extension.
    file_extension (str): Extension of the chunk files.
  '''
  from joblib import dump
  import os
  if teacher_mode :
    parts_directory = "Trained_Model_ForTeacher/"
  dump(model_list, parts_directory+file_base+file_extension)
  split_file(file_base, parts_directory, file_extension)
  os.remove(parts_directory+file_base+file_extension)
