# Initiation aux data sciences - Rakuten Data Challenge 

## Description
Ce projet est une proposition de solution au Rakuten Data Challenge.
Le descriptif du challenge est disponible ici : https://challengedata.ens.fr/professors/challenges/35/
L'objectif est de mettre en oeuvre différentes méthodes de Machine Learning et Deep Learning pour résoudre le problème

## Table des Matières
- [Installation](#installation)
- [Usage](#usage)
- [Auteurs](#auteurs)

## Installation
1. Clonez le dépôt : `git clone https://github.com/MatteoLucas/Rakuten_Data_Challenge.git`
2. Allez dans le répertoire du projet : `cd Rakuten_Data_Challenge`
3. Installez les dépendances : `npm install` ou `pip install -r requirements.txt`

## Usage
Le code permet de faire différentes choses :
### Entrainement d'un modèle
Pour entrainer un modèle il suffit de lancer dans le repertoire du projet :
```bash
python ./model.py teacher_mode
```
En remplaçant : 
    - `model` par le nom du modèle à entrainer : `SVC`, `Random_Forest`, `knn`, `ReseauNeurones`
    - `teacher_mode` par `True` ou `False` : en mode `True` le modèle sera entrainé sur la totalité de `X_train.csv` alors qu'en mode `False`, une partie de `X_train.csv` sera conservée pour les tests.

Par exemple :
```bash
python ./knn.py False
python ./SVC.py True
```
### Prédiction
Pour effectuer une prediction à partir d'un modèle entrainé, il suffit de lancer dans le repertoire du projet :
```bash
python ./predict.py model teacher_mode
```
En remplaçant : 
    - `model` par le nom du modèle à partir duquel faire la prediction : `svc`, `rf`, `knn`, `rn`
    - `teacher_mode` par `True` ou `False` : en mode `True` le modèle de prédiction sera celui entrainé sur la totalité de `X_train.csv`, un fichier `Predictions_ForTeacher/Y_pred_model.csv` sera créé alors qu'en mode `False`, le modèle de prédiction sera celui entrainé sur une partie de `X_train.csv` et le programme renvera uniquement le score f1.

## Auteurs
- Mattéo Lucas
- Mathias Polverino
- Hugo Lelièvre