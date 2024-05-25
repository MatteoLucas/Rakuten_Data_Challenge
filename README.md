# Initiation aux data sciences - Rakuten Data Challenge

## Description
Ce projet est une proposition de solution au Rakuten Data Challenge. Le descriptif du challenge est disponible ici : https://challengedata.ens.fr/challenges/35/  
L'objectif est de mettre en œuvre différentes méthodes de Machine Learning et Deep Learning pour résoudre le problème.

## Table des Matières
- [Installation](#installation)
- [Usage](#usage)
- [Résultats](#résultats)
- [Auteurs](#auteurs)

## Installation
1. Clonez le dépôt : `git clone https://github.com/MatteoLucas/Rakuten_Data_Challenge.git`  
2. Allez dans le répertoire du projet : `cd Rakuten_Data_Challenge`  
3. Installez les dépendances : `npm install` ou `pip install -r requirements.txt`  

## Usage
Le code permet de faire différentes choses :
### Entrainement d'un modèle
Pour entrainer un modèle, il suffit de lancer dans le répertoire du projet :
```bash
python Train_Functions/model.py teacher_mode
```
En remplaçant : 
- `model` par le nom du modèle à entrainer : `svm`, `rf`, `knn`, `rn`, `gb`.
- `teacher_mode` par `True` ou `False` : en mode `True`, le modèle sera entrainé sur la totalité de `X_train.csv`, alors qu'en mode `False`, une partie de `X_train.csv` sera conservée pour les tests.  

Par exemple :
```bash
python Train_Functions/knn.py False
python Train_Functions/svm.py True
```
#### Entrainement avec réduction de la dimension
Il est possible d'entraîner un modèle avec une réduction de la dimension par la méthode SVD.  
Pour ce faire, il faut tout d'abord entraîner le modèle de réduction de la dimension :
```bash
python Train_Functions/svd.py --n_components n
```
Où `n_components` est un argument facultatif qui permet de définir la nouvelle dimension. La valeur par défaut est `n_components=30000`.  

Par exemple :
```bash
python Train_Functions/svd.py --n_components 60000
```
Maintenant, pour entraîner un modèle avec une réduction de la dimension, il suffit de rajouter `--svd True` à la fin de la commande.  
Par exemple :
```bash
python Train_Functions/svm.py True --svd True
```
Dans cet exemple, le modèle créé sera nommé `svd_svm`, c'est le nom qu'il faudra utiliser pour la prédiction.
### Prédiction
Pour effectuer une prédiction à partir d'un modèle entraîné, il suffit de lancer dans le répertoire du projet :
```bash
python ./predict.py model teacher_mode
```
En remplaçant : 
- `model` par le nom du modèle à partir duquel faire la prédiction : `svm`, `rf`, `knn`, `rn`, `gb`.
- `teacher_mode` par `True` ou `False` : en mode `True`, le modèle de prédiction sera celui entraîné sur la totalité de `X_train.csv`, un fichier `Predictions_ForTeacher/Y_pred_model.csv` sera créé, alors qu'en mode `False`, le modèle de prédiction sera celui entraîné sur une partie de `X_train.csv` et le programme renverra uniquement le score f1.  

Par exemple :
```bash
python ./predict.py rf True
```

### Vote majoritaire
Il est aussi possible d'effectuer un vote majoritaire entre plusieurs modèles entraînés. Pour ce faire, il suffit de lancer dans le répertoire du projet : 
```bash
python ./vote.py model1 model2 model3 teacher_mode
```
En remplaçant : 
- `model1`, `model2`, `model3` par le nom des modèles à partir desquels faire le vote : `svm`, `rf`, `knn`, `rn`, `gb`. Il faut mettre au minimum 2 modèles.
- `teacher_mode` par `True` ou `False` : en mode `True`, les modèles de prédiction seront ceux entraînés sur la totalité de `X_train.csv`, un fichier `Predictions_ForTeacher/Y_pred_vote.csv` sera créé, alors qu'en mode `False`, les modèles de prédiction seront ceux entraînés sur une partie de `X_train.csv` et le programme renverra uniquement le score f1.  

Par exemple :
```bash
python ./vote.py svm rn rf knn True
```

## Résultats
Le tableau ci-dessous regroupe les scores obtenus par nos différents modèles lors de la soumission des résultats sur le site du challenge.

| Modèle | Abréviation | Score lors de la soumission |
|-----------|-----------|-----------|
| Support Vector Machine  | svm  | 0,8150772450094129  |
| Random Forest | rf | 0,7954135737911665 |
| Gradient Boosting | gb | 0,7857407432117608 |
| Réseau de neurones simple  | rn  | 0,7585749852167508  |
| SVM avec réduction de la dimension  | svd_svm  | 0,7463318764800209  |
| K plus proches voisins | knn | 0,7009108287785425 |  

Nous avons aussi commencé à implémenter une méthode basée sur le modèle BERT et une basée sur les CNN, mais nous avons été limités par nos capacités d'entraînement. Le travail en cours est disponible sur les branches éponymes.

## Auteurs
- Mattéo Lucas
- Mathias Polverino
- Hugo Lelièvre