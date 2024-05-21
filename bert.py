#Entraine le modèle bert et le sauvegarde
def bert_train(teacher_mode):
    
    import os
    import pandas as pd
    import torch
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from transformers import DataCollatorWithPadding
    from torch.utils.data import Dataset
    import TextProcessing as TP

    #Définir manuellement le seed pour la reproductibilité
    def set_seed(seed):
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    set_seed(42)

    # Charger les fichiers CSV et enlever la première ligne
    X_train = pd.read_csv('X_train.csv', skiprows=1, usecols=[0, 1, 2])
    Y_train = pd.read_csv('Y_train.csv', skiprows=1)

    # Renommer les colonnes pour une manipulation plus facile
    X_train.columns = ['num_produit', 'info1', 'info2']
    Y_train.columns = ['num_produit', 'categorie']

    # Fusionner les deux colonnes de texte pour l'entraînement
    X_train['texte'] = X_train['info1'].astype(str) + " " + X_train['info2'].astype(str)

    # Supprimer les lignes avec des valeurs manquantes
    X_train.dropna(subset=['texte'], inplace=True)
    Y_train.dropna(subset=['categorie'], inplace=True)

    # Fusionner les données d'entraînement pour obtenir les labels
    train_data = pd.merge(X_train[['num_produit', 'texte']], Y_train, on='num_produit')

    # Diviser les données en ensembles d'entraînement et de validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_data['texte'].astype(str).tolist(),
        train_data['categorie'].astype(int).tolist(),
        test_size=0.2,
        random_state=42
    )

    # Créer un mapping pour les labels
    unique_labels = sorted(set(train_labels + val_labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Appliquer le mapping aux labels d'entraînement et de validation
    train_labels = [label_mapping[label] for label in train_labels]
    val_labels = [label_mapping[label] for label in val_labels]

    # Vérification des labels après mapping
    num_labels = len(label_mapping)

    # Initialiser le tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Créer une classe Dataset personnalisée
    class ProductDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}  # enlever la dimension supplémentaire
            encoding['labels'] = torch.tensor(label, dtype=torch.long)
            return encoding

    # Créer des datasets d'entraînement et de validation
    train_dataset = ProductDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProductDataset(val_texts, val_labels, tokenizer)

    # Initialiser le modèle BERT pour la classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Initialiser le data collator avec padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Définir les arguments d'entraînement
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Augmenter la taille du batch si la mémoire le permet
        per_device_eval_batch_size=8,
        num_train_epochs=5,  # Ajuster le nombre d'époques
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10,
        fp16=True  # Utiliser l'entraînement en précision mixte pour accélérer l'entraînement
    )

    # Initialiser le Trainer
    bert = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: {'f1': f1_score(p.label_ids, p.predictions.argmax(-1), average='weighted')}
    )

    # Entraîner le modèle
    bert.train()

    # Sauvegarder le modèle entraîné
    TP.save_model([bert, X_train, None, Y_train, None], 'bert', teacher_mode) 

    print("Modèle entraîné et sauvegardé avec succès.")
