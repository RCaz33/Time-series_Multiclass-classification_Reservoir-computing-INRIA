Dans le cadre de l’AI 4 Industry (https://www.ai4industry.fr/), nous avons eu l’opportunité de participer au Use Case de
CATIE, centre technologique spécialisé dans la recherche, le développement et l’innovation
dans les domaines de l’information et de l’électronique.
-https://www.catie.fr/

L’objectif de ce Use Case était de développer un dispositif permettant de reconnaitre
automatiquement les chiffres tracés à l’aide d’un capteur Z motion1.
-https://6tron.io/use_case/demo-z-motion-ble-imu

Afin de pouvoir avoir un jeu de données à analyser, chacun des trois groupes a saisi, à l’aide
des capteurs, plusieurs séries de nombres allant de 0 à 9, sur tables pour certains et dans
l’espace (à l’horizontal mais aussi aléatoirement dans d’autres sens) pour d’autres. Ces
données ont été collectées via la console Linux, en initiant un enregistrement via un script
python fourni par CATIE.

Pour la Configuration 1 :
- « t » qui représente le timestamp (il n’est pas toujours régulier)
- « raw acceleration x », « raw acceleration y » et « raw acceleration z » qui représentent
les accélérations linéaires du capteur mesurées selon les axes x, y et z.
- « magnetic field x », « magnetic field y » et « magnetic field z » qui correspondent aux
champs magnétiques mesurés selon les axes x, y et z.

Pour la Configuration 3 :
- « t » qui représente le timestamp (il n’est pas toujours régulier)
- « raw acceleration x », « raw acceleration y » et « raw acceleration z » qui représentent
les accélérations linéaires du capteur mesurées selon les axes x, y et z.
- « quaternion w », « quaternion x », « quaternion y » et « quaternion z » qui eux vont
représenter les composantes d’un quaternion, une représentation mathématique d’une
orientation dans l’espace. En d’autres termes, cela va décrire l’orientation 3D du
capteur.


Les differents notebooks sont organisé :
- visualisation : 
    permet d'afficher les positions du capteur en 3 dimensions pendant l'aquisition d'un chiffre (peu prendre en compte les intervalles de temps irreguliers en moyennant l'acceleration sur 2 pas de temps).
- feature_importance : 
    Analyse ANOVA et comparaison aux coefficient (features_importance) pour les modèles scikit-learn 'LogisticRegression' et 'RandomForestClassifier' optimisés. Comparaison de la precision avec des modèles optimisés parcimonieux. (precision :0.94, recall :0.93 )
- feature_ingineering : 
    Optimisation de 3 modèles scikit-learn (LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()) par feature engineering. (score :0.96, time :0.0044)
- reservoir_computing :
    Entrainement d'un reservoir simple (model = [source >> reservoir, source] >> readout) et optimisation du nombre de neuronnes, leaking rate et spectral radius. Prediction par pas de temps (score sur tout le dataset :0.8137).
- reservoir_deep :
    Architecture de reservoir complexes (HierarchicalESN, DeepESLmodel, Multi_input)
- realtime : 
    Optimisation fine du modèle RandomForest pour l'application


Application streamlit:
1- techarger localement.
2- créer virtual environment
3- executer app.py (les requirements seront automatiquement installés)

L'application permet de selectionner une des données enregistrée et de voir pas à pas la prédiction donnée par le modèle RandomForestClassifier

![Alt text]("Capture d’écran 2024-02-21 à 22.39.20.png")
