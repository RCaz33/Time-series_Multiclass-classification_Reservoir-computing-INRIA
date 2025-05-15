# 🧠 Classification Multiclasse de Séries Temporelles avec Reservoir Computing

## 🔍 Contexte

Dans le cadre de l'initiative **AI4Industry** ([ai4industry.fr](https://www.ai4industry.fr)), nous avons participé à un Use Case proposé par le **CATIE**, centre technologique spécialisé dans la recherche, le développement et l'innovation en information et électronique.
En savoir plus : [catie.fr](https://www.catie.fr)

🎯 **Objectif** : Développer un dispositif permettant de **reconnaître automatiquement les chiffres dessinés à l'aide d'un capteur Z-Motion**.

Capteur utilisé : [Z-Motion - 6Tron](https://6tron.io/use_case/demo-z-motion-ble-imu)

---

## 🧩 Problématique

**Classification multiclasse de données de séries temporelles.**

Trois groupes ont collecté des données en traçant des chiffres de 0 à 9 à l'aide du capteur Z-Motion. Les chiffres ont été tracés en 2D (sur table/mur) et en 3D (dans l'espace). Les données ont été enregistrées via une console Linux et un script Python fourni par le CATIE.

Nous avons utilisé les **configurations 1 et 3** parmi les trois disponibles :

### 🔧 Configuration 1

* `t` : Timestamp (pas toujours régulier)
* `raw acceleration x/y/z` : Accélérations linéaires
* `magnetic field x/y/z` : Champs magnétiques

### 🔧 Configuration 3

* `t` : Timestamp
* `raw acceleration x/y/z` : Accélérations linéaires
* `quaternion w/x/y/z` : Composantes d’un quaternion, représentation mathématique de l’orientation 3D

---

## 🧪 Méthodologie

1. **Acquisition & Prétraitement**

   * Reconstruction du tracé des chiffres à partir des accélérations.
   * Compression des séries temporelles et extraction de statistiques pour la classification.

2. **Modélisation - Approches classiques**

   * Modèles testés : Logistic Regression, Decision Tree, Gradient Boosting
   * Comparaison de la précision et du rappel

3. **Modélisation - Reservoir Computing (ESN)**

   * Utilisation des séries brutes avec un **Echo State Network** (ESN)
   * Prédiction à chaque pas de temps ✨

4. **Application Streamlit**

   * Visualisation interactive des prédictions dans le temps ⏳

---

## 📁 Organisation du Projet

```bash
📁 app/
 ┣ 📁 ressources/         # Modèles, GIF, fichiers .sav
 ┣ 📄 app.py              # Application Streamlit
 ┣ 📄 utils.py            # Fonctions utilitaires
 ┣ 📄 requirements.txt    # Dépendances
 ┣ 📄 debug_app.ipynb     # Debugging notebook
📁 data/                  # Fichiers de données
📄 feature_engineering.ipynb
📄 feature_importance.ipynb
📄 Visualisation.ipynb
📄 realtime.ipynb
📄 readme.md
📄 readme_en.md
```

---

## 📓 Description des Notebooks

* **Visualisation.ipynb** 🛰️
  Affichage 3D de la trajectoire du capteur. Prise en compte de timestamps irréguliers via moyenne glissante.

* **feature\_importance.ipynb** 🧮
  Analyse ANOVA + importance des features (Logistic Regression et Random Forest).
  ✅ Précision : **0.94** | Rappel : **0.93**

* **feature\_engineering.ipynb** 🛠️
  Feature engineering et optimisation de 3 modèles.
  ✅ Score : **0.96** | Temps : **0.0044s**

* **reservoir\_computing** 🧠
  Entraînement d'un ESN simple, optimisation du nombre de neurones, leaking rate, etc.
  ✅ Score total : **0.8137**

* **reservoir\_deep** 🧬
  Architecture réservoir complexe : Hierarchical ESN, Deep ESN, entrées multiples

* **realtime.ipynb** ⚙️
  Optimisation fine du Random Forest pour l'inférence temps réel

---

## 🚀 Application Streamlit

### Lancer l'application localement :

```bash
# 1. Cloner le repo
$ git clone https://github.com/RCaz33/Time-series_Multiclass-classification_Reservoir-computing-INRIA
$ cd Time-series_Multiclass-classification_Reservoir-computing-INRIA

# 2. Créer et activer un environnement virtuel
$ python3 -m venv .venv
$ source .venv/bin/activate

# 3. Installer les dépendances et lancer l'app
$ pip install streamlit
$ streamlit run app/app.py
```

### Fonctionnalités 🖥️

* Choisir une acquisition
* Affichage pas à pas des prédictions de **RandomForestClassifier** et **ESN**

![Capture d'écran de l'application](screenshot_app.png)

---

Pour toute question ou amélioration, n'hésitez pas à ouvrir une issue ou une pull request ! 🚀
