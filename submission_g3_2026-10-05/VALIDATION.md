# Validation de la préparation G3

Date : 5 octobre 2026. Branche distante intégrée : `editorial-g3-framework`,
commits `dc71784` à `16f449f`. `origin/master` n'avait aucun nouveau commit au
fetch ; aucune PR ouverte ou fermée n'a été renvoyée par `gh pr list --state all`.
Intégration en fast-forward, sans conflit ; l'état local initial était propre.

## Résultats des contrôles

| Contrôle | Résultat | Preuve |
|---|---|---|
| Tests scientifiques | **14 passed** | `validation/pytest.txt` |
| Vérification des identités et gradients | **PASS** | `validation/ccgp_verify.txt` |
| Benchmark gelé | **90 900 lignes uniques, 101 tâches**, 7 architectures, 4 pertes | `validation/data_audit.txt` |
| Sensibilité nested-HPO | **1 080 lignes uniques** | même audit |
| Recalcul de l'estimation primaire | **+0.004788**, IC [0.000774, 0.009135], p = 0.052246 | même audit |
| Mean ranks | Ridge et GBLUP aux deux premières places pour Pearson et NDCG@10 | même audit |
| Bibliographies | Les deux fichiers `refs.bib` sont identiques | même audit |
| Manuscrit G3, une colonne, supplément | **Builds complets PASS** | `validation/build_*` |
| Lettre PDF et DOCX | **PASS** | `validation/build_cover_letter*` |
| Références LaTeX / débordements | Aucune référence/citation indéfinie, aucun Overfull après correction | logs finaux `_4.txt` |
| Abstract / article summary / running title | **215 mots / 100 mots / 36 caractères** | `validation/word_counts.json` |
| Suppléments | Chacun <2 MB, total <10 MB | `validation/packaging.txt` |
| Sources LaTeX extraites dans un répertoire vierge | **Build G3 PASS** | `validation/source_archive_build.txt` |
| Nouveau script de préparation | **Ruff PASS** | contrôle ciblé |
| Ruff global | **72 problèmes préexistants** | `validation/ruff.txt` |

Les tests ont été exécutés avec `/home/delete/venv_311/bin/python`, qui dispose
de Torch et des dépendances scientifiques. Le Python système ne dispose pas
de Torch ; son premier essai a échoué à la collecte, puis les tests ont été
relancés dans l'environnement scientifique. L'avertissement CUDA concerne la
compatibilité de cette version de Torch avec la RTX 5090 ; les tests de réseaux
ont été exécutés sur CPU. Aucune campagne GPU ni nouvelle estimation empirique
n'a été lancée. La commande numérique a rerun l'expérience A seulement ; la
variation du dernier chiffre de la corrélation lentil est sans effet sur le texte.

## Corrections de relecture

- Conditions de non-dégénérescence explicites pour l'identité de calibration.
  La pente 1 et l'intercept 0 s'appliquent au jeu d'ajustement, sans garantie
  identique sur un test indépendant.
- Distinction explicite entre les comparaisons principales sur prédictions
  brutes et l'audit brut-versus-affine de la figure 4.
- La différence de NRMSE **−0.0033** compare Pearson affine à **MSE brut** dans
  la figure principale ; **−0.0011** compare Pearson affine à **MSE affine**
  dans les analyses des pertes. Ces estimands sont différents.
- Description textuelle de la figure 4 complétée avec le panneau NDCG.
- Placeholder Zenodo retiré, références aux Supplementary Files 1–3 ajoutées.
- Contributions selon l'auteur, paragraphe IA adapté d'AOM et absence de
  conflits ajoutés dans les deux versions du manuscrit ; financement conservé.
- Lettre actualisée et calibrage de la précision numérique corrigé : tolérance
  simple précision pour la perte, float64 pour l'identité affine.
- Article summary porté à exactement 100 mots, abstract sous 250 mots.
- Débordement d'une ligne de commandes dans Supporting Information supprimé.

Contrôle visuel des pages du manuscrit G3 et de Supporting Information :
formules, auteurs, numéros de lignes/pages, figures 1–6, légendes et tables
présents ; aucune coupure ou superposition majeure observée. Figures vectorielles
conservées telles qu'intégrées par la branche éditoriale. La revue accepte une
mise en page libre pour la soumission initiale ; une éventuelle harmonisation
des lettres des panneaux et autres détails de production pourra suivre ses
instructions après acceptation.

## Limites du contrôle

Les limites scientifiques documentées dans le papier restent explicites :
HPO principal non nested, une initialisation par fold, 12 panels, architectures
adaptées plutôt que reproductions exactes, direction de désirabilité non
harmonisée, changements de classement possibles avec pente affine négative.
La relecture n'établit pas une supériorité universelle de Pearson.

Les 72 problèmes Ruff concernent le style et la qualité du code déjà présents
avant cette préparation. Ils ne sont pas masqués et n'ont pas été corrigés au
prix d'un changement de périmètre expérimental. Les builds peuvent émettre des
avertissements Underfull et un avertissement microtype/footnote sans référence
manquante ni texte débordant.

Le portail privé n'a pas été consulté : noms et ordre exacts des champs à adapter
à l'écran. Accord final des coauteurs, ORCID, numéro de convention éventuel et
choix de paiement restent des informations personnelles. Les versions IA
historiques non enregistrées ne sont pas inventées. Aucun dépôt de préprint,
aucune soumission, aucun envoi d'email et aucun dépôt Zenodo n'ont été effectués.
