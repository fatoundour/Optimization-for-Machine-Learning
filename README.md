# Optimisation pour le Machine Learning

**Optimisation numérique · Gradient accéléré de Nesterov · Choix du taux d'apprentissage des réseaux de neurones**

Ce projet académique étudie deux problématiques d'optimisation à l'aide de **Python et PyTorch** : la convergence de méthodes de gradient accéléré et l'influence de la largeur d'un réseau de neurones sur le choix du pas d'apprentissage (*learning rate*).

> 📊 **Les expériences, les graphiques et les analyses détaillées sont disponibles directement dans le notebook principal : [`homework.ipynb`](homework.ipynb).**

## Partie 1 — Gradient accéléré de Nesterov : avec et sans redémarrage

Cette partie compare deux variantes de la méthode du gradient accéléré de Nesterov sur la **fonction de Rosenbrock en dimension 2** :

- **Speed Restart** : méthode avec redémarrage de l'accélération selon un critère lié à la vitesse.
- **Vanishing Friction** : méthode sans redémarrage, avec un terme d'inertie évolutif (paramètre `b = 3`).

**Résultat présenté dans le notebook :** pour les paramètres testés, la méthode sans redémarrage progresse plus rapidement au début, mais présente des oscillations. La méthode *Speed Restart* offre un comportement plus régulier et stable. Cette conclusion concerne les expériences réalisées et ne constitue pas une supériorité générale de la méthode.

## Partie 2 — Optimisation du taux d'apprentissage d'un réseau de neurones

L'objectif est d'étudier comment adapter le **pas d'apprentissage** à la taille d'un réseau de neurones.

Le modèle étudié est un **réseau à une couche cachée avec activation ReLU**, entraîné par descente de gradient pour approximer une **fonction sinus bruitée**. La fonction de perte utilisée est l'**erreur quadratique moyenne (MSE)**.

Après une première étude pour **`d = 16`** neurones cachés, plusieurs largeurs sont explorées : **`d ∈ {5, 20, 30, 50, 80}`**.

L'objectif est d'estimer expérimentalement une loi d'échelle de la forme :

**α*(d) ≈ ν × d^γ**

L'ajustement présenté dans le notebook donne :

- **ν ≈ 0,147**
- **γ ≈ −0,472**
- Soit **α*(d) ≈ 0,147 × d^(−0,472)**

**Interprétation :** dans les conditions expérimentales étudiées, le taux d'apprentissage retenu tend à diminuer lorsque la largeur du réseau augmente. Il s'agit d'une **relation empirique**, et non d'une loi universelle applicable à tous les réseaux de neurones.

## Visualisations et résultats détaillés

📈 **Consultez [`homework.ipynb`](homework.ipynb)** pour retrouver les courbes de convergence, la comparaison des méthodes de Nesterov, les essais de taux d'apprentissage, les graphiques et les discussions méthodologiques.

L'[énoncé du projet](homework_opt_ml.pdf) précise les objectifs et les consignes des deux parties.

## Organisation du dépôt

| Fichier | Description |
| --- | --- |
| [`homework.ipynb`](homework.ipynb) | Notebook principal : implémentations, expériences, graphiques et interprétations |
| [`train_nn.ipynb`](train_nn.ipynb) | Code de référence fourni pour l'entraînement du réseau de neurones |
| [`homework_opt_ml.pdf`](homework_opt_ml.pdf) | Énoncé du projet |
| [`requirements.txt`](requirements.txt) | Bibliothèques Python nécessaires |
| [`.gitignore`](.gitignore) | Exclusion des fichiers locaux et temporaires |

## Reproduire les expériences

Créer un environnement virtuel Python, puis installer les dépendances :

```bash
python -m venv .venv
source .venv/bin/activate    # Sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Ouvrir ensuite `homework.ipynb` et exécuter les cellules dans l'ordre. Les résultats et graphiques enregistrés correspondent au travail académique initial ; **les expériences n'ont pas été réexécutées indépendamment lors de la préparation de ce dépôt**.

## Limites de l'étude

Les résultats dépendent notamment de l'initialisation, des hyperparamètres, des pas d'apprentissage testés et du budget d'entraînement. La loi d'échelle estimée décrit le cadre expérimental étudié et ne garantit pas les mêmes résultats pour d'autres architectures ou jeux de données.

## Technologies et compétences

**Python · PyTorch · NumPy · Matplotlib · Jupyter Notebook · Différentiation automatique · Optimisation numérique · Descente de gradient · Réseaux de neurones**
