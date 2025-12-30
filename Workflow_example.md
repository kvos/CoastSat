# Données et Workflow dans CoastSat

Ce document décrit :
- les **sources de données satellites** utilisées par CoastSat,
- les **étapes complètes du workflow**,
- les méthodes de **détection, analyse, correction et validation du trait de côte**.

---

## 🌍 Sources des données images

### Plateforme
- CoastSat récupère les images via **Google Earth Engine (GEE)**

### Satellites utilisés
- **Landsat**
- **Sentinel-2**

---

## 🛰️ Détails des satellites

### Landsat
- 2 **Tiers** :
  - **Tier 1** : recommandé pour les séries temporelles
  - **Tier 2** : utilisé pour l’analyse qualitative
- 2 **Collections** :
  - **Collection 2** : la plus récente (recommandée)

⚠️ Les images **Landsat 7 affectées par le Scan Line Corrector (mai 2003)** peuvent être ignorées.

---

### Sentinel-2
- Produit utilisé : **Sentinel-2 Level-1C**
- Particularité :
  - présence de **tuiles qui se chevauchent**

---

## 📥 Récupération des images

### Produits téléchargés
- **Landsat Tier 1 – Top Of Atmosphere**
- **Sentinel-2 Level-1C**

### Paramètres requis
- Région d’intérêt (ROI) :
  - définie par coordonnées
  - ou via des outils comme :
    - geojson.io
    - Google MyMaps
- Taille maximale recommandée :
  - **< 100 km²**
- Sélection :
  - plage de dates
  - missions satellites

---

## 🏖️ Détection du trait de côte

### Paramètres principaux
- Seuil maximum de couverture nuageuse
- Système de coordonnées / projection (CRS)
- Validation manuelle des shorelines (optionnelle)
- Aire minimale de plage
- Couleur dominante de la plage
- Périmètre minimal du trait de côte

---

### Shoreline de référence
- Possibilité de **dessiner manuellement une shoreline de référence**
- Utilisation :
  - éliminer les détections trop éloignées de cette référence
- Contraintes :
  - coordonnées **cartésiennes**
  - définition d’une distance maximale autorisée

---

## 📊 Analyse du trait de côte

### Séries temporelles
- Calcul de la **distance cross-shore** le long de **transects normaux** définis par l’utilisateur

### Méthode
- Intersection entre :
  - transects
  - points du trait de côte
- Calcul basé sur :
  - médiane des points
  - paramètres :
    - `min_points`
    - `max_std`
    - `max_range`
    - `min_chainage`
- Gestion :
  - intersections multiples sur un même transect

---

## 🌊 Correction de marée

### Principe
- Correction basée sur :
  - série temporelle du trait de côte
  - série temporelle de la marée
  - pente de plage

---

### Données de marée
- Utilisation de **FES2022**
  - prédiction du niveau de marée sur la ROI
- Alternative :
  - import d’une série temporelle personnalisée via fichier CSV

---

### Formule de correction
```text
correction = (tide_sat - reference_elevation) / beach_slope
cross_distance_corrected = cross_distance + correction
```



## ⏱️ Post-traitement temporel

### 🧹 Suppression des outliers

#### Élimination des artefacts
- Faux positifs
- Ombres de nuages

#### Critères de détection
- Changement maximal autorisé de la distance cross-shore
- Retour immédiat à une valeur précédente après une variation brutale
- Seuil basé sur l’indice Otsu

---

### 📊 Analyses saisonnières et mensuelles

#### Calculs
- Distance cross-shore moyenne par saison
- Distance cross-shore moyenne par mois

#### Résultats
- Tendances annuelles du trait de côte (en mètres)
- Statistiques temporelles associées

---

## 📐 Estimation de la pente de plage

### 🎯 Objectif
- Estimer la **pente de la plage** utilisée pour la correction de marée

---

### 🧪 Méthodologie

1. Sélection d’une **longue série temporelle** de shorelines
2. Conservation des dates où :
   - au moins **deux satellites** sont disponibles simultanément
3. Analyse de la série temporelle de marée
4. Définition des paramètres de recherche :
   - `min_slope`
   - `max_slope`
   - intervalles de pente
   - fréquence de marée attendue
5. Détermination de la **fréquence de Nyquist**
   - basée sur la distribution temporelle des acquisitions
6. Détection de la **fréquence de marée dominante** via :
   - transformée de **Lomb–Scargle**
7. Sélection de la pente qui :
   - minimise l’énergie du signal à la fréquence de marée dominante

➡️ **Une pente de plage est estimée pour chaque transect**

---

## ✅ Validation par données terrain

### 🧪 Méthode
- Comparaison entre :
  - séries temporelles issues de relevés terrain
  - séries temporelles générées par CoastSat

---

### 📈 Analyses
- Comparaison des distances cross-shore pour chaque transect
- Analyse de la distribution des erreurs :
  - fonctions de densité de probabilité
  - boxplots

---

## 🔁 Workflow global CoastSat

---

### 1️⃣ Installation

- Suivre les instructions du dépôt GitHub (section *Installation*)
- Recommandation :
  - utiliser **Anaconda Prompt**
- Créer :
  - un compte **Google Earth Engine (GEE)**
  - un projet déclaré **non commercial**
- Renseigner :
  - `project_name` dans les notebooks

---

### 2️⃣ Récupération des images

- Téléchargement des images via **GEE**
- Possibilité de générer :
  - animations `.gif`
- Paramétrage :
  - zone d’étude
  - plage de dates
  - missions satellites

---

### 3️⃣ Détection du trait de côte

- Définition des paramètres principaux
- Gestion :
  - couverture nuageuse
  - validation manuelle des détections
  - paramètres avancés (plage, couleur, surface minimale)
- Définition du système de projection (CRS)
- Définition optionnelle d’une **shoreline de référence**
- Exécution de CoastSat :
  - création automatique du fichier `site_name_output.pkl`
- Nettoyage initial :
  - suppression des images dupliquées
  - suppression des erreurs de géoréférencement
- Export possible :
  - format `.geojson`
- Visualisation :
  - animation GIF
  - tracé 2D des shorelines par date

---

### 4️⃣ Analyse du trait de côte

- Chargement du fichier `.pkl`
- Nettoyage et homogénéisation des données
- Définition :
  - manuelle ou automatique des transects
- Calcul :
  - distances cross-shore
- Génération :
  - séries temporelles du trait de côte

---

### 5️⃣ Correction de marée

- Récupération des séries de marée via **FES2022**
- Sauvegarde et visualisation des niveaux de marée
- Application de la correction :
```text
distance_corrigée = distance + (tide - reference_elevation) / beach_slope
```


### 6️⃣ Post-traitement temporel

- Suppression des outliers
- Vérification visuelle des séries temporelles
- Analyses saisonnières et mensuelles
- Calcul des tendances du trait de côte pour chaque transect

---

### 7️⃣ Estimation de la pente de plage

- Analyse fréquentielle de la marée
- Calcul de la pente :
  - minimisant l’énergie du signal à la fréquence de marée dominante
- Détermination :
  - d’une pente spécifique pour chaque transect

---

### 8️⃣ Validation

- Comparaison des résultats avec les données terrain
- Analyse statistique des erreurs :
  - distributions
  - indicateurs statistiques

