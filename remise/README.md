# Module : Intelligence artificielle - APP2 - Problématique

## Structure du code

- Fichier `problématique.py`  
Fichier principal contenant la génération de la représentation et l'appel aux classificateurs. Par défault, tous les graphiques des classificateurs sont affichés.  

- Fichier `ImageCollection.py`
Méthodes pertinentes ayant été ajoutées ou modifiées :
    - `generate_representation()` : génération de la représentation dans un objet ClassificationData.
    - `get_images_features()` : extraction des features pour toutes les images et normalisation.
    - `get_hsv_data(image)` : extraction des données de hue et de saturation d'une image.
    - `get_mean_max_values(image_hsv, color_index)` : calcul de la moyenne de couleurs selon le color_index. On va chercher la moyenne en x des valeurs maximales en y.
    - `hough_transform_straight_line(self, gray_img, ax=None)` : extraction des lignes dans une image
    - `categorize_hough_lines(self, lines, tolerance=3)` : categorisation des lignes en lignes verticales, horizontales et autre lignes. Retour aussi du nombre de ligne parallèles.

- Fichier `classifiers.py`
Modification et complétion de code selon les laboratoires.

- Fichier `analysis.py`
Modification et complétion de code selon les laboratoires. Modification du code pour l'utilisation de trois techniques de représentation.

## Fichiers de données utilisées

Les fichiers se trouvant dans `data\baseDeDonneesImages\` sont utilisés dans le code pour l'ensemble de données.

## Lancement du programme

Pour lancer le code à partir de la racine :
```
python problematique.py
```

