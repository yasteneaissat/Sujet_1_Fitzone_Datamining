"""Tous les chiffres choisis par le groupe, réunis au même endroit.

Il y en a trois sortes :
- les hypothèses économiques : le fichier ne les donne pas, on les suppose ;
- les données du sujet : elles viennent de l'énoncé ;
- les réglages des méthodes : chacun est justifié par un tableau produit
  par le code (voir le README).

Changer une valeur ici suffit : toute l'analyse est recalculée avec.
"""

# --- Hypothèses économiques : à remplacer par les vrais chiffres de FitZone ---
COUT_SEANCE_COACHING = 20     # € payés par FitZone pour une séance de coaching
TAUX_REUSSITE_APPEL = 0.30    # part des partants qu'un appel convainc de rester
COUT_APPEL = 5                # € par appel (temps du conseiller)
MOIS_GAGNES = 12              # mois d'abonnement gardés par un abonné retenu
NB_ABONNES_FITZONE = 10_000   # « un peu moins de 10 000 » abonnés (sujet)

# --- Données du sujet ---
NB_FINS_ENGAGEMENT = 400      # abonnés en fin d'engagement chaque mois
NB_APPELS = 80                # appels possibles chaque mois
PART_APPELEE = NB_APPELS / NB_FINS_ENGAGEMENT   # 20 % des abonnés
MOIS_PAR_AN = 12              # pour passer d'un montant mensuel à un montant annuel

# --- Réglages communs (cours « Évaluer un modèle ») ---
GRAINE = 42                   # random_state : mêmes découpages à chaque lancement
PART_TEST = 0.30              # 30 % des abonnés mis de côté pour juger un modèle
NB_TOURS = 5                  # validation croisée en 5 paquets

# --- Q2, Q5, Q6 : kNN (cours kNN, slide « Choisir k, en code ») ---
VALEURS_K_TESTEES = [1, 3, 5, 7, 11, 15, 21, 31, 51]
NOMBRE_VOISINS = 31           # à égalité d'AUC avec 21 et 51, plus stable que 21 (q5_choix_nombre_voisins.csv)

# --- Q7-Q8 : k-means (cours k-means : coude et silhouette) ---
NOMBRES_SEGMENTS_TESTES = range(1, 9)
NOMBRE_SEGMENTS = 4           # 2 selon la silhouette, 4 pour les messages marketing : le métier tranche

# --- Q9 : DBSCAN (cours DBSCAN, « Choisir le couple ε et minPts ») ---
# minPts = 2 x le nombre de colonnes (calculé dans q9_atypiques.py)
# ε = lecture du décollage de la courbe des distances au minPts-ième voisin
QUANTILE_DECOLLAGE = 0.95
EPSILONS_TESTES = [round(0.1 * i, 1) for i in range(10, 31)]   # de 1,0 à 3,0
