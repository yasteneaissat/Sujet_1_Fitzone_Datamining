"""Lecture, vérification et nettoyage des deux fichiers FitZone.

Le sujet prévient que les fichiers ne sont pas propres : valeurs manquantes,
doublons, valeurs impossibles, unités inattendues. Chaque règle de nettoyage
est une petite fonction qui note dans l'audit ce qu'elle a trouvé et la
décision prise, en une phrase. Les CSV d'origine ne sont jamais modifiés :
tout le nettoyage se fait en mémoire, à chaque lancement.
"""
import pandas as pd

from fitzone.chemins import FICHIER_ABONNES, FICHIER_SALLES

COLONNES_NUMERIQUES = [
    "anciennete_mois", "prix_mensuel_eur", "distance_domicile_m",
    "visites_par_mois", "seances_coaching", "reclamations",
]


def noter(audit, controle, valeur, decision=""):
    """Ajoute une ligne à l'audit : ce qu'on a vérifié, combien, et ce qu'on a décidé."""
    audit.append({"controle": controle, "valeur": valeur, "decision": decision})


def supprimer_doublons(abonnes, audit):
    """Supprime les lignes copiées à l'identique.

    Une ligne exportée deux fois compterait deux fois le même abonné dans
    tous les taux. On vérifie ensuite qu'aucun identifiant n'a encore deux
    lignes DIFFÉRENTES : ce serait un doublon déguisé, qu'il faudrait traiter
    à la main (le compte attendu est 0).
    """
    noter(audit, "lignes copiées à l'identique", int(abonnes.duplicated().sum()),
          "supprimées : même abonné exporté deux fois")
    abonnes = abonnes.drop_duplicates()
    noter(audit, "identifiants encore en double après suppression",
          int(abonnes["abonne_id"].duplicated().sum()), "vérification : doit valoir 0")
    return abonnes


def convertir_en_nombres(abonnes, audit):
    """Convertit les colonnes numériques (cours Pandas : pd.to_numeric).

    errors="coerce" transforme en NaN ce qui n'est pas un nombre au lieu de
    faire planter le programme ; on compte ensuite ces NaN pour le vérifier.
    """
    abonnes = abonnes.copy()
    manquants_avant = int(abonnes[COLONNES_NUMERIQUES].isna().sum().sum())
    for colonne in COLONNES_NUMERIQUES:
        abonnes[colonne] = pd.to_numeric(abonnes[colonne], errors="coerce")
    manquants_apres = int(abonnes[COLONNES_NUMERIQUES].isna().sum().sum())
    noter(audit, "valeurs non numériques dans les colonnes de nombres",
          manquants_apres - manquants_avant, "vérification : doit valoir 0")
    return abonnes


def supprimer_valeurs_impossibles(abonnes, audit):
    """Écarte les anciennetés négatives et les abonnements à 0 €.

    Une valeur physiquement impossible est une erreur de saisie : on ne peut
    pas deviner la vraie valeur, donc on retire la ligne. Une valeur seulement
    extrême (un abonné à 16 km, 9 séances de coaching) est gardée : elle est
    possible, et c'est justement l'objet de la Q9.
    """
    impossibles = (abonnes["anciennete_mois"] < 0) | (abonnes["prix_mensuel_eur"] <= 0)
    noter(audit, "ancienneté négative ou prix nul", int(impossibles.sum()),
          "lignes supprimées : erreur de saisie, valeur réelle inconnue")
    return abonnes[~impossibles].copy()


def completer_valeurs_manquantes(abonnes, audit):
    """Complète les séances de coaching et les créneaux manquants.

    - Coaching : médiane de la formule de l'abonné. Un Premium fait en médiane
      bien plus de séances qu'un Essentiel, donc la médiane générale serait
      fausse pour les deux. Médiane plutôt que moyenne car la distribution
      est asymétrique (beaucoup de 0, quelques gros consommateurs).
    - Créneau : valeur explicite « non renseigné ». Mettre le créneau le plus
      fréquent (« soir ») ajouterait des abonnés au groupe du soir alors que la
      Q1 teste justement la fidélité des abonnés du soir.
    """
    noter(audit, "séances de coaching manquantes", int(abonnes["seances_coaching"].isna().sum()),
          "remplacées par la médiane de la formule")
    mediane_de_sa_formule = abonnes.groupby("formule")["seances_coaching"].transform("median")
    abonnes["seances_coaching"] = abonnes["seances_coaching"].fillna(mediane_de_sa_formule)

    noter(audit, "créneaux manquants", int(abonnes["creneau_habituel"].isna().sum()),
          "catégorie « non renseigné » : on n'invente pas de créneau")
    abonnes["creneau_habituel"] = abonnes["creneau_habituel"].fillna("non renseigné")
    return abonnes


def preparer_colonnes(abonnes, audit):
    """Passe la distance en kilomètres et crée la cible en 0/1.

    La distance est en mètres (jusqu'à près de 16 000) : on raisonne en km.
    La cible en 0/1 permet de calculer un taux de résiliation avec une moyenne.
    """
    abonnes["distance_km"] = abonnes["distance_domicile_m"] / 1000
    noter(audit, "distance_domicile_m en mètres (maximum)", int(abonnes["distance_domicile_m"].max()),
          "convertie en kilomètres (colonne distance_km)")

    abonnes["a_resilie"] = abonnes["a_resilie"].str.strip().str.lower()
    noter(audit, "valeurs de a_resilie autres que oui / non",
          int(((abonnes["a_resilie"] != "oui") & (abonnes["a_resilie"] != "non")).sum()),
          "vérification : doit valoir 0")
    abonnes["resilie"] = (abonnes["a_resilie"] == "oui").astype(int)
    return abonnes


def fusionner_avec_salles(abonnes, salles, audit):
    """Ajoute les informations de la salle à chaque abonné (cours Pandas : merge).

    Comme au cours, on vérifie le nombre de lignes avant et après, et le
    nombre d'abonnés dont la salle n'a pas été trouvée.
    """
    donnees = pd.merge(abonnes, salles, on="salle", how="left")
    noter(audit, "lignes avant / après la fusion avec les salles", f"{len(abonnes)} / {len(donnees)}",
          "vérification : doivent être égales")
    noter(audit, "abonnés sans salle retrouvée", int(donnees["ville"].isna().sum()),
          "vérification : doit valoir 0")
    return donnees


def signaler_anciennete_incoherente(donnees, audit):
    """Repère les abonnés inscrits depuis plus longtemps que leur salle n'existe.

    Exemple : à Rueil, « ouverte depuis 9 mois », la plupart des abonnés ont
    plus de 9 mois d'ancienneté. Soit ouverte_depuis_mois n'est pas en mois,
    soit ces abonnés viennent d'une autre salle. On ne peut pas trancher avec
    le fichier : on SIGNALE, on garde les abonnés (leurs autres colonnes sont
    cohérentes) et on n'utilise pas ouverte_depuis_mois dans les analyses.
    """
    incoherents = donnees["anciennete_mois"] > donnees["ouverte_depuis_mois"]
    noter(audit, "ancienneté supérieure à l'âge de la salle", int(incoherents.sum()),
          "signalé à FitZone ; colonne ouverte_depuis_mois non utilisée")


def tableau_anciennete_par_salle(donnees):
    """Détail par salle de l'incohérence ancienneté / âge de la salle."""
    donnees = donnees.copy()
    donnees["plus_ancien_que_la_salle"] = donnees["anciennete_mois"] > donnees["ouverte_depuis_mois"]
    tableau = donnees.groupby("salle").agg(
        ville=("ville", "first"),
        ouverte_depuis_mois=("ouverte_depuis_mois", "first"),
        anciennete_max=("anciennete_mois", "max"),
        abonnes=("abonne_id", "count"),
        abonnes_plus_anciens_que_la_salle=("plus_ancien_que_la_salle", "sum"),
    )
    return tableau.sort_values("abonnes_plus_anciens_que_la_salle", ascending=False)


def charger_et_nettoyer():
    """Charge les deux fichiers, applique toutes les règles et renvoie (données, audit)."""
    abonnes = pd.read_csv(FICHIER_ABONNES, encoding="utf-8")
    salles = pd.read_csv(FICHIER_SALLES, encoding="utf-8")
    audit = []
    noter(audit, "lignes du fichier abonnés", len(abonnes))
    noter(audit, "valeurs manquantes au départ", int(abonnes.isna().sum().sum()))

    abonnes = supprimer_doublons(abonnes, audit)
    abonnes = convertir_en_nombres(abonnes, audit)
    abonnes = supprimer_valeurs_impossibles(abonnes, audit)
    abonnes = completer_valeurs_manquantes(abonnes, audit)
    abonnes = preparer_colonnes(abonnes, audit)
    donnees = fusionner_avec_salles(abonnes, salles, audit)
    signaler_anciennete_incoherente(donnees, audit)

    noter(audit, "valeurs manquantes à la fin", int(donnees.isna().sum().sum()),
          "vérification : doit valoir 0")
    noter(audit, "abonnés gardés pour l'analyse", len(donnees))
    noter(audit, "taux de résiliation", round(float(donnees["resilie"].mean()), 4))
    return donnees, pd.DataFrame(audit, dtype=object)
