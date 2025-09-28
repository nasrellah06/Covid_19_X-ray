# 🔬 Application de Détection COVID-19

Une application web Flask utilisant l'intelligence artificielle pour analyser les radiographies pulmonaires et détecter la COVID-19, la pneumonie ou des radiographies normales.

## 🚀 Fonctionnalités

- **Classification automatique** : Analyse des radiographies en 3 classes (Normal, COVID-19, Pneumonie)
- **Interface utilisateur moderne** : Design responsive et intuitif
- **Système d'authentification** : Inscription et connexion sécurisées
- **Base de données** : Stockage des utilisateurs et historique des prédictions
- **Upload d'images** : Support des formats PNG, JPG, JPEG

## 📋 Prérequis

- Python 3.8+
- MySQL 5.7+
- TensorFlow 2.12+

## 🛠️ Installation

1. **Cloner le projet**
   ```bash
   git clone <votre-repo>
   cd Covid_Flask
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration de la base de données**
   - Créer une base de données MySQL nommée `covid_bd`
   - Copier `env_example.txt` vers `.env` et configurer vos paramètres :
   ```bash
   cp env_example.txt .env
   ```

4. **Configuration du fichier .env**
   ```env
   FLASK_SECRET_KEY=votre-cle-secrete
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_USER=votre-utilisateur
   MYSQL_PASSWORD=votre-mot-de-passe
   MYSQL_DATABASE=covid_bd
   ```

## 🏃‍♂️ Lancement

```bash
python covid.py
```

L'application sera accessible sur `http://localhost:5000`

## 📁 Structure du projet

```
Covid_Flask/
├── covid.py                 # Application Flask principale
├── requirements.txt         # Dépendances Python
├── schema.sql              # Schéma de base de données
├── env_example.txt         # Exemple de configuration
├── meilleur_model_covid_RMS.keras  # Modèle IA pré-entraîné
├── templates/              # Templates HTML
│   ├── login.html
│   ├── Register.html
│   └── Covid.html
├── static/                 # Fichiers statiques
│   └── images/
├── uploads/                # Dossier d'upload des images
└── Data_test/              # Données de test
    ├── Covid/
    ├── Normal/
    └── Pneumonia/
```

## 🧠 Modèle IA

Le modèle utilisé est un réseau de neurones convolutif pré-entraîné capable de classifier les radiographies en 3 classes :
- **Normal** : Radiographie pulmonaire normale
- **COVID-19** : Signes de COVID-19
- **Pneumonie** : Pneumonie virale ou bactérienne

## 🔧 Configuration

### Variables d'environnement importantes :
- `MODEL_PATH` : Chemin vers le modèle (défaut: `meilleur_model_covid_RMS.keras`)
- `INPUT_SIZE` : Taille d'entrée des images (défaut: `224x224`)
- `CLASS_NAMES` : Noms des classes (défaut: `normal,covid,pneumonia`)

## ⚠️ Avertissement médical

Cette application est fournie à titre informatif uniquement. Elle ne remplace pas un diagnostic médical professionnel. Consultez toujours un professionnel de santé pour tout diagnostic médical.

## 📝 Utilisation

1. **Inscription** : Créez un compte utilisateur
2. **Connexion** : Connectez-vous avec vos identifiants
3. **Upload** : Téléchargez une radiographie pulmonaire
4. **Analyse** : L'IA analyse l'image et fournit un diagnostic
5. **Résultat** : Consultez le résultat avec le niveau de confiance

## 🐛 Dépannage

### Erreur de modèle
- Vérifiez que le fichier `meilleur_model_covid_RMS.keras` existe
- Vérifiez la configuration `MODEL_PATH` dans `.env`

### Erreur de base de données
- Vérifiez que MySQL est démarré
- Vérifiez les paramètres de connexion dans `.env`
- Exécutez `schema.sql` pour créer les tables

### Erreur d'upload
- Vérifiez que le dossier `uploads/` existe
- Vérifiez les permissions d'écriture

## 📊 Données de test

Le dossier `Data_test/` contient des exemples d'images pour chaque classe :
- `Covid/` : Images de radiographies COVID-19
- `Normal/` : Images de radiographies normales  
- `Pneumonia/` : Images de radiographies avec pneumonie

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer des améliorations
- Ajouter de nouvelles fonctionnalités

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.