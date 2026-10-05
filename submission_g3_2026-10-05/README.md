# Soumission G3 — 5 octobre 2026

Le paquet contient le manuscrit G3 actualisé, les trois suppléments, six figures
séparées, la lettre à l'éditeur, les textes du formulaire et les sources éditables.
Commencer par `upload/Manuscript_G3.pdf`, puis `form_fields.txt`.

## Soumettre cet après-midi

1. Se connecter sur **https://g3.msubmit.net/** et créer une nouvelle soumission.
2. Choisir **Investigation** ; sélectionner la rubrique de genomic selection /
   prediction si le menu la propose. Le libellé exact dépend du portail.
3. Copier le titre, running title, keywords, affiliations et déclarations depuis
   `form_fields.txt`. Le résumé scientifique est dans `abstract.txt` ; le résumé
   accessible de 100 mots est dans `article_summary.txt`.
4. Ajouter les auteurs dans cet ordre : **Grégory Beurier, Denis Cornet,
   Lauriane Rouan, David Cros**. Les emails CIRAD et contributions sont renseignés.
   Camille Noûs figure uniquement dans les remerciements.
5. Téléverser les fichiers selon le tableau ci-dessous. Copier `cover_letter.txt`
   dans le champ lettre, ou joindre son PDF si une pièce est demandée.
6. Compléter les seuls champs personnels restants : ORCID si demandé, éventuel
   numéro de convention, reviewers si souhaités, choix de licence et paiement.
   Voir `author_declarations_to_confirm.txt` pour les attestations finales.
7. Générer le PDF de contrôle du portail. Vérifier les quatre auteurs, les
   formules, les figures, les trois suppléments et les caractères accentués.
   Vérifier l'accord des coauteurs et l'absence de soumission concurrente avant
   de cliquer sur **Submit**. Conserver l'accusé et le numéro de manuscrit.

## Fichiers à téléverser

| Fichier dans `upload/` | Désignation dans le portail |
|---|---|
| `Manuscript_G3.pdf` | Main manuscript / Manuscript |
| `Supplementary File 1.pdf` | Supplementary material : Supporting Information |
| `Supplementary File 2.zip` | Supplementary material : analysis and simulation code |
| `Supplementary File 3.zip` | Supplementary material : source data and text alternatives |
| `Figure 1.pdf` à `Figure 6.pdf` | Figure, avec le numéro correspondant |
| `Cover_letter.pdf` | Cover letter, si demandé comme fichier |

Le manuscrit inclut déjà les figures, la Table 1 et les références. Ne pas
ajouter les figures une seconde fois dans le PDF principal. Le portail peut
accepter uniquement le manuscrit illustré et les suppléments à ce stade ; les
six PDF individuels sont disponibles si des fichiers Figure sont demandés.
Aucun graphical abstract n'est fourni. `alternatives/Manuscript_single_column.pdf`
est une autre mise en page du même contenu ; ne pas téléverser deux manuscrits.

La revue accepte une mise en page libre à la première soumission et demande les
numéros de lignes et de pages. Le PDF G3 les inclut. Les fichiers supplémentaires
restent chacun sous 2 MB et leur total sous 10 MB. Les figures sont vectorielles.
Les limites vérifiées sont 250 mots pour l'abstract, 100 pour l'article summary
et 50 caractères pour le running title. [Consignes G3](https://academic.oup.com/G3JOURNAL/pages/author-guidelines).

## Lettre et déclarations

`cover_letter.txt` contient la lettre prête à copier ; les versions PDF et DOCX
sont dans `upload/` et `source/`. Les déclarations IA et contributions sont aussi
fournies séparément dans `ai_disclosure.txt` et `author_contributions.txt`.
Le texte IA reprend AOM et a été adapté au travail présent ; il figure dans la
lettre et dans les remerciements. Il couvre Claude et Codex, avec GPT-6 pour la
préparation finale. Les versions historiques non enregistrées ne sont pas
inventées. L'OUP demande la transparence sur les usages substantiels en texte,
code ou analyse. [Politique OUP journaux](https://academic.oup.com/pages/for-authors/journals/preparing-and-submitting-your-manuscript/ai-principles-for-oxford-journals/policy-on-ai-use-and-disclosure-for-oxford-journals-authors).

Le financement Bana+/PARSADA et l'absence de conflits sont inclus. Aucun préprint
n'a été publié selon l'auteur. Aucun DOI logiciel n'a été confirmé : le DOI
EasyGeSe dans le texte appartient au dataset tiers. L'ancien placeholder Zenodo
a été supprimé. Aucune review interne simulée n'est présentée comme une review
de revue ni jointe au paquet.

## Sources et reconstruction

`source/Manuscript_LaTeX.zip` contient les sources nécessaires aux deux manuscrits
et au supplément, avec les figures, la bibliographie et la classe G3. Ce fichier
est destiné à l'édition ou à une demande de sources, pas aux suppléments publiés.
`source/Reproducibility_frozen_results.zip` conserve les résultats gelés et leur
provenance pour un éventuel dépôt public ; il dépasse la limite des suppléments.
Les datasets tiers restent accessibles par les sources citées dans l'article.

Pour rafraîchir les fichiers après une correction dans `paper/`, depuis la racine :

```bash
python3 scripts/prepare_g3_submission.py
```

La commande compile G3, la version une colonne et Supporting Information,
recrée les ZIP et la lettre, extrait les résumés et vérifie les limites. Elle
nécessite TeX Live, BibTeX, Pandoc et XeLaTeX. Elle ne relance pas les expériences.
Recopier toute correction de déclarations dans `form_fields.txt` si nécessaire.

## Validation et provenance

Voir `VALIDATION.md` et `validation/` : tests, vérifications numériques, audit des
résultats, logs LaTeX, décompte des mots et contrôle des fichiers.
`SHA256SUMS.txt` permet de vérifier les fichiers du dossier :

```bash
cd submission_g3_2026-10-05
sha256sum -c SHA256SUMS.txt
```

Les modifications de `origin/editorial-g3-framework` ont été intégrées en
fast-forward dans `master` (six commits, dernier commit `16f449f`). Aucune PR
n'existait. Les corrections de préparation ajoutent les contributions et
conflits, la déclaration IA, les références aux suppléments, une précision sur
la calibration et les descriptions textuelles de la nouvelle figure 4.
