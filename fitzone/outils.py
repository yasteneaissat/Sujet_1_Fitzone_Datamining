"""Fonctions partagées par plusieurs questions.

On les met ici pour que tout le monde calcule de la même façon :
mêmes variables, même modèle kNN, même validation croisée, mêmes graphiques.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fitzone.chemins import DOSSIER_FIGURES, DOSSIER_TABLEAUX
from fitzone.hypotheses import COUT_APPEL, GRAINE, MOIS_GAGNES, NB_TOURS, TAUX_REUSSITE_APPEL

# Validation croisée du cours : 5 paquets, chacun garde la même part de partants.
VALIDATION_CROISEE = StratifiedKFold(NB_TOURS, shuffle=True, random_state=GRAINE)

# Variables utilisées pour prédire la résiliation (Q2, Q5, Q6).
# Le créneau et le prix sont écartés : la Q1 montre qu'ils n'apportent rien
# une fois la formule connue.
VARIABLES_RESILIATION = [
    "anciennete_mois", "visites_par_mois", "reclamations", "distance_km",
    "seances_coaching", "formule_essentiel", "formule_premium",
]

# Variables qui décrivent le profil d'un abonné (Q7, Q8, Q9).
# La résiliation n'en fait pas partie : on cherche qui ressemble à qui,
# on ne regarde les départs qu'après coup.
VARIABLES_PROFIL = [
    "anciennete_mois", "prix_mensuel_eur", "distance_km",
    "visites_par_mois", "seances_coaching", "reclamations",
]


# ---------- Préparation ----------

def creer_variables_resiliation(donnees):
    """Renvoie les variables du kNN, avec la formule en colonnes 0/1.

    Le kNN calcule des distances, il ne sait pas lire « Premium ».
    Comme au TP d'exploration : (colonne == valeur).astype(int).
    Confort sert de référence : un abonné Confort a 0 dans les deux colonnes.
    """
    variables = donnees[["anciennete_mois", "visites_par_mois", "reclamations",
                         "distance_km", "seances_coaching"]].copy()
    variables["formule_essentiel"] = (donnees["formule"] == "Essentiel").astype(int)
    variables["formule_premium"] = (donnees["formule"] == "Premium").astype(int)
    return variables[VARIABLES_RESILIATION]


def taux_par_groupe(donnees, colonne):
    """Taux de résiliation et nombre d'abonnés pour chaque valeur d'une colonne.

    La moyenne d'une colonne 0/1 est un taux. observed=True : on ne garde que
    les tranches qui contiennent des abonnés (utile après un pd.cut).
    """
    return donnees.groupby(colonne, observed=True).agg(
        taux_resiliation=("resilie", "mean"),
        abonnes=("abonne_id", "count"),
    )


# ---------- kNN ----------

def creer_modele_knn(nombre_voisins):
    """kNN précédé d'une standardisation, dans un seul pipeline.

    La standardisation est obligatoire : sans elle, l'ancienneté (0 à 96)
    écraserait les réclamations (0 à 4) dans le calcul des distances.
    Le pipeline la recalcule sur les seules lignes d'apprentissage à chaque
    tour de validation croisée : sinon, fuite de données (cours kNN).
    """
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=nombre_voisins))


def auc_validation_croisee(variables, cible, nombre_voisins):
    """AUC moyenne et écart-type sur les 5 tours de validation croisée.

    L'AUC juge le classement (cours « Évaluer un modèle ») : la chance que le
    modèle donne plus de risque à un partant qu'à un fidèle tiré au hasard.
    0,5 = hasard. C'est la bonne mesure ici, puisqu'on veut CLASSER les
    abonnés pour appeler les premiers. On annonce toujours moyenne ± écart-type.
    """
    scores = cross_val_score(creer_modele_knn(nombre_voisins), variables, cible,
                             cv=VALIDATION_CROISEE, scoring="roc_auc")
    return scores.mean(), scores.std()


def mesurer_appels(cible, probabilites, part):
    """Ce que donne l'appel de la part la plus à risque des abonnés.

    Le seuil n'est pas 0,50 : c'est la capacité d'appel qui le fixe (80 sur
    400 = 20 %). On trie les abonnés du score le plus haut au plus bas, on
    « appelle » les premiers, puis on compte comme au cours : les quatre
    nombres de la matrice de confusion, la précision et le rappel.
    À score égal, l'ordre du fichier départage (kind="stable").
    """
    tableau = pd.DataFrame({"partant": list(cible), "score": list(probabilites)})
    tableau = tableau.sort_values("score", ascending=False, kind="stable")
    nombre_appeles = int(len(tableau) * part)
    appeles = tableau.head(nombre_appeles)
    trouves = int(appeles["partant"].sum())
    partants = int(tableau["partant"].sum())
    return {
        "appeles": nombre_appeles,
        "trouves": trouves,                                   # vrais positifs
        "appeles_pour_rien": nombre_appeles - trouves,        # faux positifs
        "partis_sans_appel": partants - trouves,              # faux négatifs
        "laisses_tranquilles": len(tableau) - nombre_appeles - (partants - trouves),  # vrais négatifs
        "precision": trouves / nombre_appeles,     # parmi les appelés, la part de partants
        "rappel": trouves / partants,              # parmi tous les partants, la part trouvée
        "seuil": appeles["score"].min(),           # score du dernier abonné appelé
    }


# ---------- Argent ----------

def gain_campagne_appels(partants_contactes, nombre_appels, prix_moyen):
    """Gain d'une campagne d'appels : valeur des départs évités moins coût des appels.

    Même forme qu'au cours (gain = valeur x trouvés - coût x signalés), avec
    valeur d'un partant trouvé = taux de réussite x prix x mois gardés.
    Toutes les hypothèses sont dans hypotheses.py.
    """
    abonnes_retenus = partants_contactes * TAUX_REUSSITE_APPEL
    return abonnes_retenus * prix_moyen * MOIS_GAGNES - nombre_appels * COUT_APPEL


# ---------- Enregistrement ----------

def sauvegarder_tableau(tableau, nom_fichier, index=True):
    """Enregistre un tableau en CSV lisible dans Excel (utf-8-sig garde les accents).

    os.makedirs crée le dossier s'il n'existe pas encore (exist_ok=True :
    pas d'erreur s'il existe déjà).
    """
    os.makedirs(DOSSIER_TABLEAUX, exist_ok=True)
    tableau.to_csv(f"{DOSSIER_TABLEAUX}/{nom_fichier}", encoding="utf-8-sig", index=index)


def sauvegarder_graphique(figure, nom_fichier):
    """Enregistre un graphique en PNG puis le ferme pour libérer la mémoire."""
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)
    figure.savefig(f"{DOSSIER_FIGURES}/{nom_fichier}", dpi=150, bbox_inches="tight")
    plt.close(figure)


def tracer_taux_resiliation(axe, taux, taux_global, titre):
    """Barres de taux de résiliation avec la moyenne en pointillés rouges."""
    axe.bar(taux.index.astype(str), taux.values * 100, color="steelblue")
    axe.axhline(taux_global * 100, color="red", linestyle="--", label="moyenne")
    axe.set_title(titre)
    axe.set_ylabel("Résiliations (%)")
    axe.tick_params(axis="x", rotation=30)
