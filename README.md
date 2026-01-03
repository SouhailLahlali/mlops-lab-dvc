# Étape 1 : Initialisation de DVC dans le projet

Initialiser DVC à la racine du projet :

```bash
dvc init
```

![Initialisation DVC](https://github.com/user-attachments/assets/963cca27-706f-44ba-9981-3fd307f6f4e7)

---

# Étape 2 : Versionner les données brutes avec DVC

## Instructions

### 1. Supprimer `data/` du fichier `.gitignore`

Supprimer le dossier `data/` du fichier `.gitignore` afin de permettre à DVC de gérer les données.

### 2. Versionner le fichier de données

```bash
dvc add data/raw.csv
```

![Ajout du fichier avec DVC](https://github.com/user-attachments/assets/dcdec11b-8c50-47bf-b7a8-fd0bc69ceb5d)

### 3. Fichiers générés automatiquement

* `data/raw.csv.dvc`
* `data/.gitignore`

![Fichiers générés par DVC](https://github.com/user-attachments/assets/5ffdc340-29eb-4e37-9eff-9e901e195de7)

### 4. Ajouter les fichiers au versionnement Git

```bash
git add data/raw.csv.dvc data/.gitignore .gitignore
git commit -m "data: suivi du dataset brut via DVC"
```

### Résultat attendu

* `data/raw.csv` n’est plus suivi par Git
* seul `raw.csv.dvc` est versionné
* Git reste léger
* DVC gère les fichiers volumineux

![Résultat attendu](https://github.com/user-attachments/assets/dff6fd9b-640d-4685-b78f-64369907402c)

---

# Étape 3 : Configuration d’un remote DVC

```bash
mkdir dvc_storage
s# Déclare le remote principal
 dvc remote add -d localremote dvc_storage
# Pour un cloud storage : dvc remote add -d storage s3://mybucket/dvcstore
```

Versionne la configuration :

```bash
git add .dvc/config
git commit -m "dvc: configuration du remote local"
```

---

# Étape 4 : Push des données dans le remote DVC

```bash
dvc push
```

Vérifie que le dossier `dvc_storage/` contient des fichiers de hash.

![Push DVC](https://github.com/user-attachments/assets/cf854aad-51bc-4ac9-bc64-0af53c4a563a)
![Dossier remote](https://github.com/user-attachments/assets/f461e666-45c6-4cab-af29-a329af755ff8)

Résultat attendu :

La donnée est désormais partagée et récupérable par n’importe quel étudiant via DVC.

---

# Étape 5 : Simulation d’une collaboration

Supprime le dataset local :

```bash
# Windows
del data\raw.csv
# Linux/Mac
rm data/raw.csv
```

Vérifie qu’il n’y a plus rien :

```bash
ls data/
```

Récupère le dataset via DVC :

```bash
dvc pull
```

Résultat attendu :

Le fichier `data/raw.csv` réapparaît identique à l’original.

![Récupération DVC](https://github.com/user-attachments/assets/60973caa-44b3-4c22-980d-9433c4379915)

---

# Étape 6 : Création d’un pipeline reproductible dvc.yaml

### 1. Étape Prepare

```bash
dvc stage add -n prepare \
  -d src/prepare_data.py \
  -d data/raw.csv \
  -o data/processed.csv \
  -o registry/train_stats.json \
  python src/prepare_data.py
```

### 2. Étape Train

```bash
dvc stage add -n train \
  -d src/train.py -d data/processed.csv \
  -o models/model.joblib \
  python src/train.py
```

### 3. Étape Evaluate

```bash
dvc stage add -n evaluate \
  -d src/evaluate.py \
  -d models/model.joblib \
  -d data/processed.csv \
  -o reports/metrics.json \
  python src/evaluate.py
```

Versionne :

```bash
git add dvc.yaml registry/.gitignore reports/.gitignore
git commit -m "pipeline: ajout des étapes prepare/train/evaluate"
```

Résultat attendu :

DVC a enregistré dans `dvc.yaml` :

* les dépendances
* les sorties
* les commandes du pipeline

![Pipeline prepare](https://github.com/user-attachments/assets/aebcebee-1cf9-4c85-a5fc-b07c18a3bd4d)
![Pipeline train](https://github.com/user-attachments/assets/6787675f-042e-4380-a5b9-5b3b3600fed8)
![Pipeline evaluate](https://github.com/user-attachments/assets/661e185e-7f00-4008-81ab-c7fbbb2f8b6f)
![Pipeline overview](https://github.com/user-attachments/assets/d1338c80-07fd-418a-84fe-e0cb68bf346b)

---

# Étape 7 : Reproduire automatiquement tout le pipeline

Modifie par exemple `src/prepare_data.py` (ajouter un print, changer un seuil…) et vérifie l’impact :

```bash
dvc repro
```

Versionne :

```bash
git add dvc.lock
git commit -m "pipeline: lock after repro"
```

Résultat attendu :

Seules les étapes impactées sont réexécutées → reproductibilité totale.

![Repro pipeline 1](https://github.com/user-attachments/assets/c09e71ad-1f4f-48d2-9341-ea0ab5a45a59)
![Repro pipeline 2](https://github.com/user-attachments/assets/ebaeca64-53b1-40ee-9375-472cdeebd4eb)
