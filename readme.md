Dans le cadre de l’AI 4 Industry (https://www.ai4industry.fr/), nous avons eu l’opportunité de participer au Use Case de
CATIE, centre technologique spécialisé dans la recherche, le développement et l’innovation
dans les domaines de l’information et de l’électronique.
-https://www.catie.fr/

L’objectif de ce Use Case était de développer un dispositif permettant de reconnaitre
automatiquement les chiffres tracés à l’aide d’un capteur Z motion1.
-https://6tron.io/use_case/demo-z-motion-ble-imu

Problématique : Classification multiclasse de series temporelles 

Afin de pouvoir avoir un jeu de données à analyser, trois different groupes ont saisi, à l’aide
des capteurs, plusieurs séries de nombres allant de 0 à 9, en 2D sur tables/murs et en 3D dans
l’espace. Les données ont été collectées via la console Linux mise à disposition par CATIE en initiant un script
python. Il existe trois mode d'aquisitions, nous avons utilisé les configurations 1 et 3.

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

Méthodologie : Après aquisition des données, le tracé des chiffres est reconstitué à partir des accelerations sur 3 axes. Les séries temporelles sont compréssées et leur valeurs statistiques sont utilisées à des fin de classification multiclasse en comparant la précisions pour la regression logisitique, un arbre de decision et un un gradient boosting classifier. Ensuite les series temporelles brutes sont utilisées et chaque pas de temps est classifié avec un modèle echo state network (ESN: un type de réservoir computing qui utilise un réseau neuronal récurrent avec une couche cachée peu connectée). Finalement, une application streamlit est mise à dispoition pour visualiser le processus de decision de chacune des aquisitions de nombres au cours du temps.

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
1- techarger localement (git clone https://github.com/RCaz33/Time-series_Multiclass-classification_Reservoir-computing-INRIA).
2- créer virtual environment (python3 venv .venv; source .venv/bin/activate)
3- installer streamlit et executer app.py (pip install streamlit; streamlit run app.py)

L'application permet de selectionner une des données enregistrée et de voir pas à pas la prédiction donnée par les modèle RandomForestClassifier et ESN entrainés sur tous les datasets.

![Alt text](screenshot_app.png)
