# Examen DVC et DagsHub

Mise en place d'un workflow de modélisation reproductible avec **DVC** et **DagsHub**
sur le jeu de données de flottation minérale.

La variable cible est `silica_concentrate`. Le jeu de données brut contient
1817 observations, 8 variables explicatives et une colonne `date` (non utilisée
pour la modélisation).

## Structure du dépôt

```
examen-dvc
├── data
│   ├── raw_data              # données brutes suivies par DVC
│   │   └── raw.csv
│   ├── processed_data        # sorties de la pipeline (split + normalisation)
│   └── prediction.csv        # prédictions du modèle (sortie DVC)
├── metrics
│   └── scores.json           # métriques d'évaluation (suivies par Git)
├── models                    # scaler, meilleurs paramètres et modèle entraîné (DVC)
├── src
│   ├── data
│   │   ├── download_data.py  # téléchargement du dataset brut
│   │   ├── split_data.py     # split train / test
│   │   └── normalize_data.py # standardisation
│   └── models
│       ├── grid_search.py    # recherche des meilleurs hyperparamètres
│       ├── train_model.py    # entraînement du modèle final
│       └── evaluate_model.py # évaluation, métriques et prédictions
├── dvc.yaml                  # définition de la pipeline DVC
├── dvc.lock                  # verrouillage des hashs de la pipeline
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Données

Le dataset brut est téléchargé puis suivi par DVC :

```bash
python src/data/download_data.py
dvc add data/raw_data/raw.csv
```

## Pipeline DVC

La pipeline enchaîne cinq étapes :

```text
split → normalize → grid_search → train → evaluate
```

```bash
dvc repro          # exécute les étapes nécessaires
dvc dag            # affiche le graphe de la pipeline
dvc metrics show   # affiche les métriques d'évaluation
```

Les étapes sont déclarées dans `dvc.yaml` et verrouillées dans `dvc.lock`.
Elles peuvent aussi être recréées avec `dvc stage add` :

```bash
dvc stage add -n split \
  -d src/data/split_data.py -d data/raw_data/raw.csv \
  -o data/processed_data/X_train.csv -o data/processed_data/X_test.csv \
  -o data/processed_data/y_train.csv -o data/processed_data/y_test.csv \
  python src/data/split_data.py

dvc stage add -n normalize \
  -d src/data/normalize_data.py \
  -d data/processed_data/X_train.csv -d data/processed_data/X_test.csv \
  -o data/processed_data/X_train_scaled.csv \
  -o data/processed_data/X_test_scaled.csv \
  -o models/scaler.pkl \
  python src/data/normalize_data.py

dvc stage add -n grid_search \
  -d src/models/grid_search.py \
  -d data/processed_data/X_train_scaled.csv -d data/processed_data/y_train.csv \
  -o models/best_params.pkl \
  python src/models/grid_search.py

dvc stage add -n train \
  -d src/models/train_model.py \
  -d data/processed_data/X_train_scaled.csv -d data/processed_data/y_train.csv \
  -d models/best_params.pkl \
  -o models/trained_model.pkl \
  python src/models/train_model.py

dvc stage add -n evaluate \
  -d src/models/evaluate_model.py -d models/trained_model.pkl \
  -d data/processed_data/X_test_scaled.csv -d data/processed_data/y_test.csv \
  -o data/prediction.csv -M metrics/scores.json \
  python src/models/evaluate_model.py
```

## Modèle

Le modèle est un `GradientBoostingRegressor`. Le GridSearch teste :

| Paramètre | Valeurs |
|-----------|---------|
| `n_estimators` | 100, 200 |
| `learning_rate` | 0.05, 0.1 |
| `max_depth` | 3, 5 |
| `subsample` | 0.8, 1.0 |

La sélection se fait en validation croisée 5 folds avec le score R².
Le scaler est ajusté uniquement sur l'ensemble d'entraînement pour éviter
toute fuite de données.

Derniers résultats sur le jeu de test :

| MSE | RMSE | MAE | R² |
|-----|------|-----|----|
| 0.759 | 0.871 | 0.667 | 0.242 |

## Stockage distant DagsHub

DagsHub utilise un stockage S3. Après avoir créé le dépôt DagsHub lié au dépôt
GitHub, récupérer les identifiants depuis le bouton **Remote → DVC** et lancer :

```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<utilisateur>/<depot>.s3
dvc remote modify origin --local access_key_id <access_key_id>
dvc remote modify origin --local secret_access_key <secret_access_key>
dvc remote default origin
dvc push
```

Les identifiants sont stockés dans `.dvc/config.local`, qui n'est pas versionné.

## Rendu

- Partager le dépôt DagsHub avec `https://dagshub.com/licence.pedago` en
  collaborateur avec droits de lecture.
- Envoyer sur la plateforme un `.zip` contenant un `.md` avec nom, prénom,
  adresse e-mail et lien vers le dépôt DagsHub.

Checklist du dépôt :

- [x] 5 scripts de preprocessing, modélisation et évaluation
- [x] `.dvc/config` avec l'emplacement distant
- [x] modèle `.pkl` suivi par DVC dans le dossier `models`
- [x] `metrics/scores.json`
- [x] `dvc.yaml` et `dvc.lock`
- [x] données visibles dans l'onglet *Data* de DagsHub
- [x] schéma de pipeline généré par DagsHub
