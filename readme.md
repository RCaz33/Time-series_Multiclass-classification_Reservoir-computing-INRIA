Dans le cadre de l’AI 4 Industry, nous avons eu l’opportunité de participer au Use Case de
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
- feature importance : 
- feature ingineering :
- reservoir_computing
- reservoir_deep
- realtime