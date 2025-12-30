# Guide Complet : Adaptation de CoastSat pour la Détection des Côtes Rocheuses

## Table des matières

1. [Contexte et Objectif](#1-contexte-et-objectif)
2. [Architecture Actuelle de CoastSat](#2-architecture-actuelle-de-coastsat)
3. [Nouvelle Architecture Proposée](#3-nouvelle-architecture-proposée)
4. [Étapes d'Implémentation](#4-étapes-dimplémentation)
5. [Modifications du Code](#5-modifications-du-code)
6. [Entraînement du Nouveau Classifieur](#6-entraînement-du-nouveau-classifieur)
7. [Tests et Validation](#7-tests-et-validation)
8. [Annexes](#8-annexes)

---

## 1. Contexte et Objectif

### 1.1 Problématique

CoastSat est actuellement optimisé pour détecter le trait de côte à l'interface **eau/sable**. Cette approche est insuffisante pour :

- **Falaises rocheuses** : pas de plage sableuse visible
- **Côtes rocheuses** : interface eau/roche directe
- **Plateformes d'abrasion** : zones rocheuses intermittentes

### 1.2 Objectif

Modifier CoastSat pour détecter le trait de côte à l'interface **eau/roche** en ajoutant une 5ème classe "Rock" au classifieur existant.

### 1.3 Approche retenue

| Aspect | Avant | Après |
|--------|-------|-------|
| Nombre de classes | 4 | 5 |
| Classes | Sand, Whitewater, Water, Other | Sand, Whitewater, Water, **Rock**, Other |
| Interface détectée | Eau/Sable | Eau/Sable **OU** Eau/Roche |
| Indices spectraux | MNDWI, NDWI, NIR-R | + **NDBI**, **RI** (Rock Index) |

---

## 2. Architecture Actuelle de CoastSat

### 2.1 Workflow de détection

```
┌─────────────────────────────────────────────────────────────────────┐
│                        extract_shorelines()                          │
│                       (SDS_shoreline.py:41)                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    1. Prétraitement de l'image                       │
│                    SDS_preprocess.preprocess_single()                │
│                    → im_ms (5 bandes: B, G, R, NIR, SWIR)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    2. Classification des pixels                      │
│                    classify_image_NN() (ligne 326)                   │
│                    → 4 classes: Sand, Whitewater, Water, Other      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. Calcul des features                            │
│                    calculate_features() (ligne 263)                  │
│                    → 20 features (bandes + indices + std)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. Extraction des contours                        │
│                    find_wl_contours2() (ligne 442)                   │
│                    → Interface eau/sable via MNDWI                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5. Post-traitement                                │
│                    process_shoreline() (ligne 619)                   │
│                    → Filtrage et conversion coordonnées             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Classification actuelle (4 classes)

| Label | Classe | Description | Couleur |
|-------|--------|-------------|---------|
| 0 | Other | Végétation, bâtiments, roches (non distinguées) | Jaune |
| 1 | Sand | Sable de plage | Orange |
| 2 | Whitewater | Écume, zone de déferlement | Cyan |
| 3 | Water | Eau profonde | Bleu |

### 2.3 Indices spectraux actuels

```python
# Dans calculate_features() - SDS_shoreline.py:263-324

# Bandes multispectrales (im_ms)
# im_ms[:,:,0] = Blue
# im_ms[:,:,1] = Green
# im_ms[:,:,2] = Red
# im_ms[:,:,3] = NIR (Near Infrared)
# im_ms[:,:,4] = SWIR (Short Wave Infrared)

# Indices calculés :
NDWI  = (NIR - Green) / (NIR + Green)    # Ligne 294 : im_NIRG
MNDWI = (SWIR - Green) / (SWIR + Green)  # Ligne 297 : im_SWIRG
NDVI  = (NIR - Red) / (NIR + Red)        # Ligne 300 : im_NIRR (proche NDVI)
       = (SWIR - NIR) / (SWIR + NIR)     # Ligne 303 : im_SWIRNIR
       = (Blue - Red) / (Blue + Red)     # Ligne 306 : im_BR
```

### 2.4 Fichiers clés

| Fichier | Fonction | Rôle |
|---------|----------|------|
| `coastsat/SDS_shoreline.py` | `extract_shorelines()` | Workflow principal |
| `coastsat/SDS_shoreline.py` | `classify_image_NN()` | Classification pixels |
| `coastsat/SDS_shoreline.py` | `calculate_features()` | Calcul indices spectraux |
| `coastsat/SDS_shoreline.py` | `find_wl_contours2()` | Extraction contours eau/sable |
| `coastsat/SDS_classify.py` | `label_images()` | Labellisation interactive |
| `coastsat/SDS_tools.py` | `nd_index()` | Calcul indices normalisés |
| `classification/train_new_classifier.ipynb` | - | Entraînement classifieur |

---

## 3. Nouvelle Architecture Proposée

### 3.1 Nouvelle classification (5 classes)

| Label | Classe | Description | Couleur proposée |
|-------|--------|-------------|------------------|
| 0 | Other | Végétation, bâtiments, sols nus | Jaune |
| 1 | Sand | Sable de plage | Orange |
| 2 | Whitewater | Écume, zone de déferlement | Cyan |
| 3 | Water | Eau | Bleu |
| **4** | **Rock** | **Roches, falaises, plateformes rocheuses** | **Marron/Gris** |

### 3.2 Nouveaux indices spectraux à ajouter

#### NDBI - Normalized Difference Built-up Index
```
NDBI = (SWIR - NIR) / (SWIR + NIR)
```
- **Valeurs élevées** : zones minérales, roches, bâtiments
- **Valeurs faibles** : végétation, eau
- **Intérêt** : Les roches ont généralement un NDBI plus élevé que le sable

#### RI - Rock Index (proposé)
```
RI = (Red - Green) / (Red + Green)
```
- **Intérêt** : Les roches ont souvent une réflectance plus élevée dans le rouge que le vert

#### BSI - Bare Soil Index
```
BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
```
- **Intérêt** : Discrimine les sols nus et les surfaces minérales

### 3.3 Signatures spectrales typiques

```
                    Blue    Green   Red     NIR     SWIR
Eau profonde        0.05    0.04    0.02    0.01    0.00
Eau turbide         0.08    0.07    0.05    0.03    0.01
Sable sec           0.20    0.25    0.30    0.35    0.40
Sable humide        0.10    0.12    0.15    0.18    0.22
Roche sombre        0.05    0.06    0.08    0.12    0.15
Roche claire        0.15    0.18    0.22    0.28    0.32
Végétation          0.03    0.05    0.03    0.40    0.15
```

### 3.4 Workflow modifié

```
┌─────────────────────────────────────────────────────────────────────┐
│                    2. Classification des pixels                      │
│                    classify_image_NN_5classes()  ← NOUVELLE FONCTION│
│                    → 5 classes: Sand, Whitewater, Water, Rock, Other│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    3. Calcul des features                            │
│                    calculate_features_extended() ← NOUVELLE FONCTION│
│                    → 26 features (+ NDBI, RI, BSI + std)            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ Si pixels "Sand" détectés    │  │ Si pixels "Rock" détectés    │
│ find_wl_contours2()          │  │ find_wl_contours_rock()      │
│ → Interface eau/sable        │  │ → Interface eau/roche        │
└──────────────────────────────┘  └──────────────────────────────┘
                       │                       │
                       └───────────┬───────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Fusion des shorelines                             │
│                    → Trait de côte complet                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Étapes d'Implémentation

### Vue d'ensemble

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 : Préparation                                                  │
│  ├── Étape 1.1 : Créer une branche de développement                    │
│  ├── Étape 1.2 : Sélectionner les sites d'entraînement rocheux         │
│  └── Étape 1.3 : Télécharger les images des sites rocheux              │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 2 : Modification du code                                         │
│  ├── Étape 2.1 : Modifier calculate_features() - ajouter NDBI, RI, BSI │
│  ├── Étape 2.2 : Modifier classify_image_NN() - 5 classes              │
│  ├── Étape 2.3 : Modifier label_images() - ajouter classe Rock         │
│  ├── Étape 2.4 : Créer find_wl_contours_rock()                         │
│  └── Étape 2.5 : Modifier extract_shorelines() - logique hybride       │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 3 : Entraînement                                                 │
│  ├── Étape 3.1 : Labelliser des images de côtes rocheuses              │
│  ├── Étape 3.2 : Fusionner avec les données d'entraînement existantes  │
│  └── Étape 3.3 : Entraîner et sauvegarder le nouveau classifieur       │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 4 : Tests et validation                                          │
│  ├── Étape 4.1 : Tester sur des côtes rocheuses connues                │
│  ├── Étape 4.2 : Tester sur des côtes sableuses (non-régression)       │
│  └── Étape 4.3 : Évaluer la précision et ajuster les paramètres        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Modifications du Code

### 5.1 Modifier `calculate_features()` dans `SDS_shoreline.py`

**Fichier** : `coastsat/SDS_shoreline.py`
**Fonction** : `calculate_features()` (ligne 263)
**Objectif** : Ajouter les indices NDBI, RI et BSI

#### Code actuel (lignes 263-324) :
```python
def calculate_features(im_ms, cloud_mask, im_bool):
    # ... code existant ...
    # NIR-G (NDWI)
    im_NIRG = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G (MNDWI)
    im_SWIRG = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # ... etc ...
```

#### Code modifié :
```python
def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification.
    The features include spectral normalized-difference indices and standard
    deviation of the image for all the bands and indices.

    MODIFIED: Added NDBI, RI and BSI indices for rock detection.

    KV WRL 2018 - Modified for rocky coastline detection 2024

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool

    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool,0],axis=1)
    for k in range(1,im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool,k],axis=1)
        features = np.append(features, feature, axis=-1)

    # ===== INDICES EXISTANTS =====
    # NIR-G (NDWI)
    im_NIRG = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool],axis=1), axis=-1)
    # SWIR-G (MNDWI)
    im_SWIRG = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool],axis=1), axis=-1)
    # NIR-R
    im_NIRR = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool],axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool],axis=1), axis=-1)
    # B-R
    im_BR = SDS_tools.nd_index(im_ms[:,:,0], im_ms[:,:,2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool],axis=1), axis=-1)

    # ===== NOUVEAUX INDICES POUR LA DÉTECTION DES ROCHES =====

    # NDBI - Normalized Difference Built-up Index (SWIR - NIR) / (SWIR + NIR)
    # Valeurs élevées pour les zones minérales/rocheuses
    im_NDBI = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,3], cloud_mask)
    features = np.append(features, np.expand_dims(im_NDBI[im_bool],axis=1), axis=-1)

    # RI - Rock Index (Red - Green) / (Red + Green)
    # Les roches ont souvent une réflectance plus élevée dans le rouge
    im_RI = SDS_tools.nd_index(im_ms[:,:,2], im_ms[:,:,1], cloud_mask)
    features = np.append(features, np.expand_dims(im_RI[im_bool],axis=1), axis=-1)

    # BSI - Bare Soil Index ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
    # Discrimine les sols nus et surfaces minérales
    im_BSI_num = (im_ms[:,:,4] + im_ms[:,:,2]) - (im_ms[:,:,3] + im_ms[:,:,0])
    im_BSI_den = (im_ms[:,:,4] + im_ms[:,:,2]) + (im_ms[:,:,3] + im_ms[:,:,0])
    im_BSI = np.divide(im_BSI_num, im_BSI_den,
                       out=np.zeros_like(im_BSI_num),
                       where=im_BSI_den!=0)
    im_BSI[cloud_mask] = np.nan
    features = np.append(features, np.expand_dims(im_BSI[im_bool],axis=1), axis=-1)

    # ===== ÉCARTS-TYPES DES BANDES =====
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std = SDS_tools.image_std(im_ms[:,:,k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    # ===== ÉCARTS-TYPES DES INDICES EXISTANTS =====
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    # ===== ÉCARTS-TYPES DES NOUVEAUX INDICES =====
    im_std = SDS_tools.image_std(im_NDBI, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_RI, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BSI, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool],axis=1), axis=-1)

    return features
```

**Nombre de features** :
- Avant : 20 features (5 bandes + 5 indices + 5 std bandes + 5 std indices)
- Après : 26 features (5 bandes + 8 indices + 5 std bandes + 8 std indices)

---

### 5.2 Modifier `classify_image_NN()` dans `SDS_shoreline.py`

**Fichier** : `coastsat/SDS_shoreline.py`
**Fonction** : `classify_image_NN()` (ligne 326)
**Objectif** : Ajouter la classe "Rock" (label = 4)

#### Code modifié :
```python
def classify_image_NN(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 5 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - rock (rocky coastline, cliffs, platforms)     --> label = 4  # NOUVEAU
        - other (vegetation, buildings...)              --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018 - Modified for rocky coastline detection 2024

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        4D image containing a boolean image for each class (im_classif == label)
        Order: (sand, swash, water, rock)

    """

    # calculate features
    vec_features = calculate_features(im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool))
    vec_features[np.isnan(vec_features)] = 1e-9 # NaN values are create when std is too close to 0

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0]*cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_inf = np.any(np.isinf(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud,np.logical_or(vec_nan,vec_inf))
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan*np.ones((cloud_mask.shape[0]*cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    im_rock = im_classif == 4  # NOUVEAU

    # remove small patches of sand, water or rock that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(im_sand, min_size=min_beach_area, connectivity=2)
    im_water = morphology.remove_small_objects(im_water, min_size=min_beach_area, connectivity=2)
    im_rock = morphology.remove_small_objects(im_rock, min_size=min_beach_area, connectivity=2)  # NOUVEAU

    # Stack avec 4 couches : sand, swash, water, rock
    im_labels = np.stack((im_sand, im_swash, im_water, im_rock), axis=-1)

    return im_classif, im_labels
```

---

### 5.3 Créer `find_wl_contours_rock()` dans `SDS_shoreline.py`

**Fichier** : `coastsat/SDS_shoreline.py`
**Objectif** : Nouvelle fonction pour extraire l'interface eau/roche

#### Nouvelle fonction à ajouter après `find_wl_contours2()` :
```python
def find_wl_contours_rock(im_ms, im_labels, cloud_mask, im_ref_buffer):
    """
    Method for extracting rocky shorelines. Uses the classification
    to identify rock pixels and finds the water/rock interface.

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        4D image containing a boolean image for each class in the order (sand, swash, water, rock)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu water/rock threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:,:,3], im_ms[:,:,1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows*ncols, 2)

    # reshape labels into vectors - utiliser rock (index 3) au lieu de sand (index 0)
    vec_rock = im_labels[:,:,3].reshape(ncols*nrows)   # Rock est maintenant à l'index 3
    vec_water = im_labels[:,:,2].reshape(ncols*nrows)  # Water reste à l'index 2

    # use im_ref_buffer and dilate it by 5 pixels
    se = morphology.disk(5)
    im_ref_buffer_extra = morphology.binary_dilation(im_ref_buffer, se)
    # create a buffer around the rocky area
    vec_buffer = im_ref_buffer_extra.reshape(nrows*ncols)

    # select water/rock pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer, vec_water),:]
    int_rock = vec_ind[np.logical_and(vec_buffer, vec_rock),:]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_rock) > 0:
        if np.argmin([int_rock.shape[0], int_water.shape[0]]) == 1:
            int_rock = int_rock[np.random.choice(int_rock.shape[0], int_water.shape[0], replace=False),:]
        else:
            int_water = int_water[np.random.choice(int_water.shape[0], int_rock.shape[0], replace=False),:]

    # threshold the rock/water intensities
    int_all = np.append(int_water, int_rock, axis=0)
    t_mwi = filters.threshold_otsu(int_all[:,0])
    t_wi = filters.threshold_otsu(int_all[:,1])

    # find contour with Marching-Squares algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)
    # remove contour points that are NaNs (around clouds)
    contours_wi = process_contours(contours_wi)
    contours_mwi = process_contours(contours_mwi)

    # only return MNDWI contours and threshold
    return contours_mwi, t_mwi
```

---

### 5.4 Modifier `extract_shorelines()` pour la logique hybride

**Fichier** : `coastsat/SDS_shoreline.py`
**Fonction** : `extract_shorelines()` (ligne 41)
**Objectif** : Utiliser soit eau/sable, soit eau/roche selon les pixels détectés

#### Modifications dans la boucle principale (vers ligne 191-205) :

```python
# Dans extract_shorelines(), remplacer la section de détection des contours par :

else:
    try: # use try/except structure for long runs
        # Compter les pixels sand et rock dans le buffer de référence
        n_sand_pixels = sum(im_labels[im_ref_buffer, 0]) if im_labels.shape[2] > 0 else 0
        n_rock_pixels = sum(im_labels[im_ref_buffer, 3]) if im_labels.shape[2] > 3 else 0

        # Décider quelle méthode utiliser selon le type de côte dominant
        if n_sand_pixels >= 50 and n_sand_pixels > n_rock_pixels:
            # Côte sableuse : utiliser la méthode classique eau/sable
            contours_mwi, t_mndwi = find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer)

        elif n_rock_pixels >= 50:
            # Côte rocheuse : utiliser la nouvelle méthode eau/roche
            contours_mwi, t_mndwi = find_wl_contours_rock(im_ms, im_labels, cloud_mask, im_ref_buffer)

        else:
            # Pas assez de pixels classifiés : utiliser la méthode traditionnelle
            # compute MNDWI image (SWIR-G)
            im_mndwi = SDS_tools.nd_index(im_ms[:,:,4], im_ms[:,:,1], cloud_mask)
            # find water contours on MNDWI grayscale image
            contours_mwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)

    except:
        print('Could not map shoreline for this image: ' + filenames[i])
        continue
```

---

### 5.5 Modifier `label_images()` dans `SDS_classify.py`

**Fichier** : `coastsat/SDS_classify.py`
**Fonction** : `label_images()` (ligne 70)
**Objectif** : Ajouter l'interface pour labelliser les pixels rocheux

#### Ajouter une nouvelle section après "digitize land pixels" (vers ligne 309) :

```python
##############################################################
# digitize rock pixels (with lassos) - NOUVELLE SECTION
##############################################################
color_rock = settings['colors']['rock']
ax.set_title('Click and hold to draw lassos and select ROCK pixels\nwhen finished press <Enter>')
fig.canvas.draw_idle()
selector_rock = SelectFromImage(ax, implot, color_rock)
key_event = {}
while True:
    fig.canvas.draw_idle()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.waitforbuttonpress()
    if key_event.get('pressed') == 'enter':
        selector_rock.disconnect()
        break
    elif key_event.get('pressed') == 'escape':
        selector_rock.array = im_sand_ww_water_land  # nom de variable à adapter
        implot.set_data(selector_rock.array)
        fig.canvas.draw_idle()
        selector_rock.implot = implot
        selector_rock.im_bool = np.zeros((selector_rock.array.shape[0], selector_rock.array.shape[1]))
        selector_rock.ind=[]
# update im_viz and im_labels
im_viz = selector_rock.array
selector_rock.im_bool = selector_rock.im_bool.astype(bool)
im_labels[selector_rock.im_bool] = settings['labels']['rock']
```

---

### 5.6 Modifier les settings dans le notebook d'entraînement

**Fichier** : `classification/train_new_classifier.ipynb`
**Cellule** : Settings (cellule 2)

```python
settings = {
    'filepath_train': filepath_train,
    'cloud_thresh': 0.9,
    'cloud_mask_issue': True,
    'pan_off': False,
    'inputs': {'filepath': filepath_images},
    # Labels avec 5 classes maintenant
    'labels': {
        'sand': 1,
        'white-water': 2,
        'water': 3,
        'rock': 4,              # NOUVEAU
        'other land features': 0
    },
    # Couleurs pour la visualisation
    'colors': {
        'sand': [1, 0.65, 0],           # Orange
        'white-water': [1, 0, 1],       # Magenta
        'water': [0.1, 0.1, 0.7],       # Bleu
        'rock': [0.5, 0.35, 0.2],       # Marron  # NOUVEAU
        'other land features': [0.8, 0.8, 0.1]  # Jaune
    },
    'tolerance': 0.01,
    's2cloudless_prob': 60,
}
```

---

### 5.7 Modifier la fonction `show_detection()` pour afficher les roches

**Fichier** : `coastsat/SDS_shoreline.py`
**Fonction** : `show_detection()` (ligne 715)

Modifier la section de création de l'image classifiée :

```python
# compute classified image - MODIFIÉ pour 5 classes
im_class = np.copy(im_RGB)
cmap = plt.get_cmap('tab20c')
colorpalette = cmap(np.arange(0,13,1))
colours = np.zeros((4,4))  # Maintenant 4 couleurs (sand, swash, water, rock)
colours[0,:] = colorpalette[5]                    # Sand - orange
colours[1,:] = np.array([204/255,1,1,1])          # Whitewater - cyan
colours[2,:] = np.array([0,91/255,1,1])           # Water - bleu
colours[3,:] = np.array([139/255,90/255,43/255,1]) # Rock - marron (NOUVEAU)

for k in range(0, im_labels.shape[2]):
    im_class[im_labels[:,:,k],0] = colours[k,0]
    im_class[im_labels[:,:,k],1] = colours[k,1]
    im_class[im_labels[:,:,k],2] = colours[k,2]

# Modifier aussi la légende :
orange_patch = mpatches.Patch(color=colours[0,:], label='sand')
white_patch = mpatches.Patch(color=colours[1,:], label='whitewater')
blue_patch = mpatches.Patch(color=colours[2,:], label='water')
brown_patch = mpatches.Patch(color=colours[3,:], label='rock')  # NOUVEAU
black_line = mlines.Line2D([],[],color='k',linestyle='-', label='shoreline')
ax2.legend(handles=[orange_patch, white_patch, blue_patch, brown_patch, black_line],
           bbox_to_anchor=(1, 0.5), fontsize=10)
```

---

## 6. Entraînement du Nouveau Classifieur

### 6.1 Sélection des sites d'entraînement rocheux

Créer des fichiers `.kml` pour des sites avec des côtes rocheuses. Exemples recommandés :

| Site | Pays | Type de côte | Fichier |
|------|------|--------------|---------|
| Étretat | France | Falaises calcaires | `ETRETAT.kml` |
| Biarritz | France | Rochers basaltiques | `BIARRITZ.kml` |
| Côte de Granit Rose | France | Granite | `GRANIT_ROSE.kml` |
| Pointe du Raz | France | Falaises | `POINTE_RAZ.kml` |
| Cassis | France | Calanques calcaires | `CASSIS.kml` |

### 6.2 Workflow d'entraînement

```python
# 1. Télécharger les images
for site in train_sites_rocky:
    polygon = SDS_tools.polygon_from_kml(os.path.join(filepath_sites, site))
    polygon = SDS_tools.smallest_rectangle(polygon)
    sitename = site[:site.find('.')]
    inputs = {'polygon': polygon, 'dates': dates, 'sat_list': sat_list,
              'sitename': sitename, 'filepath': filepath_images}
    metadata = SDS_download.retrieve_images(inputs)

# 2. Labelliser les images (avec la nouvelle classe 'rock')
for site in train_sites_rocky:
    settings['inputs']['sitename'] = site[:site.find('.')]
    metadata = SDS_download.get_metadata(settings['inputs'])
    SDS_classify.label_images(metadata, settings)

# 3. Charger les labels
features = SDS_classify.load_labels(train_sites_rocky, settings)

# 4. Fusionner avec les données existantes (optionnel mais recommandé)
with open(os.path.join(filepath_train, 'CoastSat_training_set_L8.pkl'), 'rb') as f:
    features_original = pickle.load(f)

# Ajouter les nouvelles données de roches
for key in features.keys():
    if key in features_original:
        features[key] = np.append(features[key], features_original[key], axis=0)

# 5. Formater et entraîner
classes = ['sand', 'white-water', 'water', 'rock', 'other land features']
labels = [1, 2, 3, 4, 0]
X, y = SDS_classify.format_training_data(features, classes, labels)

# 6. Entraîner le classifieur
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
classifier = MLPClassifier(hidden_layer_sizes=(100, 50), solver='adam')
classifier.fit(X_train, y_train)
print('Accuracy: %0.4f' % classifier.score(X_test, y_test))

# 7. Sauvegarder le nouveau modèle
joblib.dump(classifier, os.path.join(filepath_models, 'NN_5classes_Landsat_rock.pkl'))
```

### 6.3 Nombre de pixels recommandé par classe

| Classe | Minimum | Recommandé | Maximum |
|--------|---------|------------|---------|
| Sand | 1000 | 5000 | 10000 |
| Whitewater | 500 | 1000 | 3000 |
| Water | 1000 | 5000 | 10000 |
| **Rock** | **1000** | **5000** | **10000** |
| Other | 1000 | 5000 | 10000 |

---

## 7. Tests et Validation

### 7.1 Sites de test recommandés

| Site | Type | Attendu |
|------|------|---------|
| Soulac-sur-Mer | Sable | Non-régression |
| Étretat | Falaises | Détection roches |
| Biarritz | Mixte | Détection hybride |

### 7.2 Métriques de validation

```python
# Calculer la matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred,
                           target_names=['other', 'sand', 'whitewater', 'water', 'rock']))

# Visualiser la matrice de confusion
SDS_classify.plot_confusion_matrix(y_test, y_pred,
                                   classes=['other', 'sand', 'whitewater', 'water', 'rock'],
                                   normalize=True)
```

### 7.3 Critères de succès

| Métrique | Seuil minimum | Objectif |
|----------|---------------|----------|
| Accuracy globale | > 95% | > 98% |
| Precision (rock) | > 85% | > 90% |
| Recall (rock) | > 85% | > 90% |
| F1-score (rock) | > 85% | > 90% |

---

## 8. Annexes

### 8.1 Résumé des fichiers à modifier

| Fichier | Modifications |
|---------|---------------|
| `coastsat/SDS_shoreline.py` | `calculate_features()`, `classify_image_NN()`, `find_wl_contours_rock()`, `extract_shorelines()`, `show_detection()` |
| `coastsat/SDS_classify.py` | `label_images()` |
| `classification/train_new_classifier.ipynb` | Settings, workflow |

### 8.2 Nouvelles dépendances

Aucune nouvelle dépendance requise. Toutes les bibliothèques utilisées (numpy, scikit-learn, scikit-image) sont déjà présentes.

### 8.3 Rétrocompatibilité

Pour maintenir la compatibilité avec les anciens modèles à 4 classes :

```python
# Dans extract_shorelines(), détecter le nombre de classes du modèle
n_classes = len(np.unique(clf.classes_))

if n_classes == 4:
    # Ancien modèle 4 classes
    im_classif, im_labels = classify_image_NN(im_ms, cloud_mask, min_beach_area_pixels, clf)
else:
    # Nouveau modèle 5 classes
    im_classif, im_labels = classify_image_NN_5classes(im_ms, cloud_mask, min_beach_area_pixels, clf)
```

### 8.4 Arborescence des nouveaux fichiers

```
CoastSat/
├── coastsat/
│   ├── SDS_shoreline.py          # Modifié
│   └── SDS_classify.py           # Modifié
├── classification/
│   ├── models/
│   │   ├── NN_4classes_Landsat.pkl        # Existant
│   │   ├── NN_5classes_Landsat_rock.pkl   # NOUVEAU
│   │   └── NN_5classes_S2_rock.pkl        # NOUVEAU
│   ├── training_data/
│   │   ├── CoastSat_training_set_L8.pkl   # Existant
│   │   └── CoastSat_training_set_rock.pkl # NOUVEAU
│   └── training_sites/
│       ├── ETRETAT.kml                    # NOUVEAU
│       └── BIARRITZ.kml                   # NOUVEAU
└── README_ROCKY_COASTLINE.md              # Ce fichier
```

---

## Prochaines étapes

1. **Créer une branche Git** pour le développement
2. **Implémenter `calculate_features()`** avec les nouveaux indices
3. **Tester sur une image** pour valider les nouveaux indices
4. **Labelliser des images** de côtes rocheuses
5. **Entraîner le nouveau classifieur**
6. **Tester et valider** sur plusieurs sites

---

*Document créé pour le projet CoastSat - Détection des côtes rocheuses*
*Version 1.0*
