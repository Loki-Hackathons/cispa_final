# Task 1 — Roadmap score (0.349 → objectif 0.42+ public)

**Rédigé :** 2026-07-03, ~01h30. **État de départ :** public **0.349** (#994, `b50_ps2_elo_entbin5_iso`, CV 0.4301).
**Contexte complet :** [`attempt1.md`](attempt1.md). **Métrique :** [`scoring.md`](scoring.md).

> **MISE À JOUR 2026-07-03, ~08h30 — TOUT EST IMPLÉMENTÉ ET TESTÉ (unitairement).**
> Les workstreams A, B, C1, C2, D et l'ensemble sont codés, compilés, et validés par
> `test_roadmap_units.py` (20 checks OK) + `test_no_adjacent.py` (régression forward-backward
> intacte). **Il ne reste que l'exécution**, qui exige les `.npz` GPU absents de cette machine
> (poste Bastian) — lancer sur le poste d'Alexandre ou JURECA. Récap des commandes : §"Exécution"
> tout en bas. Nouveaux fichiers : `gen_synth.py`, `validate_synth.py`, `run_synth.sh`,
> `ensemble_smm.py`, `test_roadmap_units.py`. Fichiers modifiés : `smm_scorer.py`, `fit_smm.py`,
> `cv_smm.py` (rétro-compatibles : les configs existantes produisent des scores identiques,
> vérifié par tests d'identité).

**Règle d'or (inchangée) :** toute décision se prend sur le **CV 5-fold document-level** (`cv_smm.py`), pas sur val seule. On ne soumet que si le CV bat 0.4301 sur **seed 0 ET seed 1**. Historique : les gains CV transfèrent à ~75–100 % au public ; les réglages val-only ne transfèrent pas.

---

## 0. Où sont les points perdus (chiffres de l'audit §17, base #771/#994)

C'est la carte des gisements — chaque workstream ci-dessous attaque une ligne précise :

| Gisement | Chiffre actuel | Levier | Workstream |
| -------- | -------------- | ------ | ---------- |
| **KGW quasi aveugle** | **9 %** de spans KGW détectés (vs 62 % GM, 44 % TS), alors que KGW ≈ 22 % des spans | LLR KGW en forme fermée via logits 7B (`kgw_lpg`) — **codé, npz prêts, jamais évalué** | **A (P0)** |
| Grille non re-balayée depuis le passage 7B | entbin4/6/7, b40/b60, ps3, evlo définis dans `CONFIGS` mais évalués seulement avec l'entropie proxy 0.5B | relancer la grille de raffinement avec l'entropie 7B | **B (P0)** |
| Spans courts sans signal | 34 % des spans (long. moyenne 73) à score ≈ bruit — pur effet √L, tables binnées bruitées à petit échantillon | plus de données de fit → **données synthétiques** (on a les clés + le générateur + le vendor code) | **D (P1-GPU)** |
| Spans en bordure de doc | 11 % détectés vs 31 % internes (88 spans concernés) | hypothèses tronquées structurées par longueur canonique | **C (P1)** |
| FP = imprécision de frontière | 3/4 des clusters de FP à 1–5 tokens d'un vrai span | érosion/post-processing des frontières du posterior | **C (P1)** |
| Écart de niveau CV→public (~75 %) | distributionnel, pas de l'overfit (§12) | recalibration H0 semi-supervisée sur le test | **E (P2)** |

Budget FPR pour intuition : test ≈ 1,32 M tokens, ~0,9 M clean → **~900 FP autorisés** au total (~270 sur le public 30 %). Chaque token clean sur-noté coûte cher.

---

## Workstream A (P0) — LLR KGW/Unigram exacts via logits 7B — **à lancer immédiatement**

**Pourquoi c'est le levier n°1 :** le Bernoulli global KGW (taux vert H1 fitté, une seule valeur ou 5 par bin d'entropie) écrase toute l'information de position. `kgw_lpg` (calculé au §18, jamais branché en éval) donne, **par position**, la probabilité verte sous la distribution boostée du vrai générateur — le LLR en forme fermée `green_exact_llr` corrige exactement le biais de confiance (déjà vérifié : corr(vert réalisé, p_green boosté) = 0.69 même sur tokens clean, et g0 moyen = 0.2507 ≈ gamma). Contrairement à la vraisemblance exacte Gumbel/TextSeal (négative, §18), **le mécanisme ne dépend pas de temperature/top-p** : le ratio boosté/non-boosté est invariant à ces params au premier ordre (le boost est appliqué aux logits avant softmax dans les deux mondes).

**Estimation de gain :** KGW ≈ 22 % des spans, détection 9 % → si on atteint ne serait-ce que le niveau TextSeal (44 %), ≈ +7 pts de détection span-level → ordre de grandeur **+0.03 à +0.06 CV**. Même un demi-succès bat tous les micro-raffinements restants.

**Prérequis :** `output/{kgw_lpg,unigram_lpg}_{train,validation,test}.npz` (produits par le job `15401646`, §18) + les npz kgw/entropy habituels.

**Commandes (CPU local, ~100 s/config) :**

```bash
cd cispa_final/task_1_text_watermark/alexandre

# 1. les 4 variantes déjà définies dans CONFIGS, seed 0
python cv_smm.py --configs b50_ps2_elo_entbin5_iso_kgwx b50_ps2_elo_entbin5_iso_unix \
                           b50_ps2_elo_entbin5_iso_bothx b50_ps2_elo_entbin5_iso_allx \
    | tee cv_lpg_seed0.log

# 2. si kgwx > 0.4301 : valider seed 1 (éditer SEED=1 dans cv_smm.py ou paramétrer)
# 3. si confirmé : --final + soumission
python cv_smm.py --final <winner> --out submission_kgwx.jsonl
python ../../shared/submit.py submission_kgwx.jsonl \
  --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "SMM entbin5+iso + closed-form KGW LLR from 7B boosted green-mass"
```

**Si `kgwx` déçoit (< baseline) :** avant d'abandonner, tester 2 variantes rapides — (a) `boost` ≠ 1.5 est exclu (YAML), mais **mixer** l'émission `kgw_exact` AVEC le `bernoulli_ent` existant comme hypothèses alternatives (ajouter une config où les deux coexistent au lieu de remplacer) ; (b) baisser `exact_clip` à 4. Le pattern "exact remplace → dilue" a déjà été observé sur Gumbel (§18) — la version **ensemble** est la bonne assurance.

**Go/no-go :** 2 h max sur ce workstream. Décision : gain reproductible sur 2 seeds → soumettre ; sinon documenter dans `attempt1.md` et passer à B.

---

## Workstream B (P0, parallélisable avec A) — re-balayer la grille avec l'entropie 7B

Les hyperparams du winner actuel (`entbin5 × 50 bins`) ont été sélectionnés **avec l'entropie proxy 0.5B**. L'entropie 7B est nettement plus informative (+16.7 % rel.) — l'optimum de la grille a probablement bougé (plus de bins d'entropie devient peut-être payant maintenant que le signal d'entropie est propre).

**Commandes (~25 min la passe) :**

```bash
python cv_smm.py --configs b40_ps2_elo_entbin4 b40_ps2_elo_entbin5 b50_ps2_elo_entbin4 \
                           b50_ps2_elo_entbin6 b60_ps2_elo_entbin5 b50_ps3_elo_entbin5 \
                           b50_ps2_evlo_entbin5 b30_ps2_elo_entbin7 | tee cv_regrid_7b.log
```

Ajouter à `CONFIGS` (3 lignes chacune) et tester aussi, **toutes avec `isotonic_llr=True`** (validé, ne coûte rien) :

- `entbin8 × 50` et `entbin10 × 40` — pousser le conditionnement entropie plus loin ;
- `clip=6` sur le winner (le clip 4 date de la grille 30-bins) ;
- sweep fit : `ASSIGN_Z` 2.5 / 3.5 et `MIN_SPAN_FIT` 10 (constantes de `fit_smm.py`, jamais balayées — pureté vs taille des pools H1) ;
- `pseudo=0.5` et `pseudo=2.0` dans `fit_binned_llr` (lissage des bins de queue, interagit avec l'isotonic).

**Rendement attendu :** +0.005 à +0.015 CV. Pas de soumission dédiée sauf si > +0.01 sur 2 seeds — sinon on l'embarque dans la prochaine soumission d'un autre workstream.

---

## Workstream C (P1) — bordures : spans tronqués + érosion des frontières

Deux constats d'audit distincts, deux fixes locaux (CPU, dans `smm_scorer.py`) :

**C1. Spans tronqués structurés par longueur.** Aujourd'hui les hypothèses préfixe `[0,e)` / suffixe `[s,n)` utilisent un prior plat (`edge_prior`) sans structure de longueur. Or un span tronqué au bord est un span **canonique coupé** : la partie visible de longueur ℓ a pour prior `Σ_L p(L) · P(coupe donne ℓ | L)`. Implémenter ce prior (au lieu du plat) devrait relever les 88 spans de bord (11 % → cible 20 %+). Effort : ~1 h. Test : config `_edgeprior` en CV.

**C2. Érosion des frontières.** Les FP qui fixent le seuil sont des débordements de 1–5 tokens autour de vrais spans. Le posterior est déjà exact par token — mais la fenêtre de longueur candidate "mal calée" étale la masse. Post-processing candidat, à tester tel quel (10 lignes, hors modèle) : `score'_t = min(score_{t-1}, score_t, score_{t+1})` (érosion morphologique rayon 1, ou rayon 2) appliqué **seulement aux tokens dont le score est dans la zone du seuil**. Coût : quelques TP aux vraies frontières (~2 tokens × spans détectés) ; gain : casse les clusters de FP de 6–10 tokens. Le trade-off est favorable si les clusters FP sont plus fréquents que les frontières exactes — c'est exactement ce que le CV tranchera. Effort : 30 min.

**Go/no-go :** chaque fix isolé en CV ; garder si +0.003 reproductible (même seuil que l'isotonic, qui a payé).

---

## Workstream D (P1, GPU — lancer le job TÔT, exploiter plus tard) — données synthétiques labellisées

**Le plus gros levier restant sur le fond.** 34 % des spans sont perdus faute de SNR, et toutes les tables (LLR binnés × entropie, taux verts KGW) sont estimées sur **180 docs / ~790 spans**. On possède tout ce qu'il faut pour générer des données d'entraînement illimitées : les clés exactes (YAML), le code de génération vendor (textseal, lm-watermarking, unigram-watermark, commits épinglés), le générateur réel (Qwen2.5-7B-Instruct, en cache sur JURECA depuis §18).

**Plan :**

1. Script `gen_synth.py` : prompts variés (reprendre les débuts des docs test comme amorces — même distribution de domaine), génération par blocs alternés clean / watermark ON avec les 4 schémas, longueurs de spans tirées du prior canonique {31,47,63,95,159,320}, labels exacts loggés à la génération. Cible : **1 000–2 000 docs** (~1 GPU·h sur A100 au vu des 14 min/1500 docs du forward pass — la génération est plus lente, compter 2–4 h, batcher).
2. **Validation de réalisme AVANT usage** (critique) : comparer les pools H1 synthétiques vs labellisés par schéma (KS-test sur les distributions de signaux, taux verts KGW par bin d'entropie). Si mismatch → les params de sampling (temperature/top-p) diffèrent de ceux des organisateurs → ne pas utiliser tel quel, essayer 2–3 réglages de sampling et garder celui qui matche.
3. Usage : **augmentation des pools de fit uniquement** (les 180 docs réels restent le held-out CV — ne jamais mettre de synthétique dans l'éval). Config `*_synth` : `fit_params(docs_reels + docs_synth, ...)`. Bénéfice attendu : tables binnées × entropie quasi sans bruit d'estimation → aide disproportionnée sur les spans courts et les bins de queue qui fixent le seuil.

**Estimation de gain :** +0.02 à +0.05 CV si le réalisme est validé ; ~0 sinon (et on le sait avant de soumettre). **Lancer la génération maintenant** pour que les données soient prêtes quand A/B/C sont épuisés.

```bash
# JURECA — après écriture de gen_synth.py + run_synth.sh (sbatch 1×A100, 4h)
sbatch run_synth.sh   # → output/synth_docs.jsonl (+ labels)
```

---

## Workstream E (P2 — seulement si A–D épuisés) — recalibration H0 sur le test

L'écart de niveau CV→public (~75 %) est distributionnel (§12). Les tables LLR côté **H0** peuvent être re-estimées sans labels : prendre les régions du test où le posterior actuel est très clean (score < p20), les traiter comme pool H0 test, re-fitter les **edges de bins + côté H0 du LLR** dessus (le côté H1 reste fitté sur le labellisé). Répétition d'une itération type EM.

**Risque :** feedback loop (les FP actuels contaminent le pseudo-pool H0). Mitigation : seuil de sélection très conservateur + valider d'abord la mécanique en CV simulé (fitter H0 sur val-sans-labels, mesurer sur val). **Ne soumettre que si le CV simulé est ≥ neutre.** Effort : 2 h. Gain espéré : spécifiquement sur le public (là où le CV ne peut pas le voir) — c'est le seul levier qui attaque directement le ratio 75 %.

---

## Ensemble final (P1, trivial, à faire avant la dernière soumission)

Moyenne des **log-odds** (pas des scores — repasser par `logit(score)` puis sigmoïde) des 2–4 meilleures configs CV distinctes (ex. winner + kgwx + regrid). Toujours ≥ le meilleur composant en général, jamais testé ici. 20 lignes de script, un run CV pour valider. À embarquer dans la soumission finale si +0.002 CV.

---

## Séquencement proposé (nuit/matinée du 3 juillet)

| Heure (indicatif) | Action | Qui/Quoi |
| ----------------- | ------ | -------- |
| T+0 | **A** : CV des 4 configs lpg (seed 0) — 10 min | CPU local |
| T+0 (parallèle) | **D** : écrire + lancer `gen_synth.py` sur JURECA | GPU |
| T+30 min | **B** : grille regrid 7B (seed 0) pendant que A tourne en seed 1 | CPU |
| T+2 h | Soumettre le meilleur de A∪B s'il bat 0.4301 ×2 seeds | API |
| T+2 h → T+5 h | **C1 + C2** (bordures), CV à chaque étape | CPU |
| T+5 h | Récupérer les données synth, valider réalisme, config `*_synth` en CV | CPU |
| T+6 h | Soumission consolidée (meilleure combinaison validée + ensemble) | API |
| Ensuite | **E** si le temps le permet et que A–D sont épuisés | CPU |
| Dernière heure | **Figer** : re-vérifier format (1320 docs, scores ∈ [0,1]), s'assurer que la meilleure soumission est bien la dernière retenue côté leaderboard | — |

**Budget soumissions :** cooldown ~5 min, pas de rareté — mais chaque soumission doit être loggée (`--method` précis) et motivée par un gain CV ×2 seeds. Pas de soumission "pour voir".

---

## Ce qu'il ne faut PAS refaire (négatifs déjà établis — ne pas y retourner)

| Piste | Verdict | Réf |
| ----- | ------- | --- |
| Vraisemblance exacte Gumbel/TextSeal (`logp_target`) | négatif partout (6 variantes), cause = sampling params inconnus | §18 |
| Pondération multiplicative par entropie | négatif (casse le drift H0) | §16 |
| Poids de mixture par prévalence | négatif | §17.C |
| Interdiction spans adjacents (`noadj`) | neutre (+0.0006), complexité non justifiée | §17.A |
| Shifts faibles / grilles denses de shifts | négatif | §11 |
| Prior de longueur uniforme/mixé | négatif | §12 |
| Unigram (scan, signal supplémentaire) | plafond physique atteint (~1 doc test) | §9 |
| Vocab KGW alternatif | écarté | §8 |

**Exception autorisée :** la vraisemblance exacte Gumbel/TextSeal peut être ré-ouverte UNIQUEMENT si le workstream D (synthétique) révèle les vrais params de sampling (le réalisme des pools synthétiques valide/invalide temperature+top-p) — dans ce cas `r | H0` redevient calculable et les configs `_exw*` (poids réduit, déjà codées) méritent un run.

---

## Récap des attentes de gain (cumulables, estimations honnêtes)

| Workstream | Gain CV estimé | Confiance |
| ---------- | -------------- | --------- |
| A — KGW lpg | +0.03 à +0.06 | moyenne (mécanisme solide, jamais mesuré) |
| B — regrid 7B | +0.005 à +0.015 | haute (rendement décroissant connu) |
| C — bordures | +0.005 à +0.01 | moyenne |
| D — synthétique | +0.02 à +0.05 | moyenne-basse (dépend du réalisme) |
| E — recalib H0 test | public only, non mesurable en CV | basse |
| Ensemble | +0.002 à +0.005 | haute |

Chemin réaliste : CV 0.43 → **0.47–0.52**, soit public **~0.38–0.42** au ratio de transfert observé (75–81 %). Chemin pessimiste (A et D échouent) : CV ~0.445 → public ~0.36.

---

## Exécution (état 2026-07-03 ~08h30 — tout est codé, il ne reste qu'à lancer)

### Ce qui a été implémenté (poste Bastian, code committé — voir diff)

| Workstream | Implémentation | Configs CV à lancer |
| ---------- | -------------- | ------------------- |
| **A** — KGW lpg | `Emission.signal` (aliasing de signal) + `fit_params(kgw_exact_ensemble=, kgw_exact_clip=)` : la LLR fermée s'ajoute **en hypothèse supplémentaire** à côté du Bernoulli fitté (pattern assurance §18), au lieu de le remplacer | `b50_ps2_elo_entbin5_iso_kgwx` (remplace), `_kgwxens` (ensemble), `_kgwx_c4`, `_kgwxens_c4`, `_unix`, `_bothx`, `_allx` |
| **B** — regrid 7B | `fit_params(assign_z=, min_span_fit=, bin_pseudo=)` exposés et branchés jusqu'à `collect_pools`/`fit_binned_llr` | `b50_ps2_elo_entbin8_iso`, `b40_ps2_elo_entbin10_iso`, `b50_ps2_elo_entbin6_iso`, `_c6`, `_az25`, `_az35`, `_msf10`, `_pse05`, `_pse2` + les configs entbin4/6, b40/b60, ps3, evlo préexistantes |
| **C1** — spans tronqués | `fit_smm.fit_edge_len_prior()` (prior P(ℓ) = Σ_L≥ℓ p(L)/L sur la longueur visible) + `SmmParams.edge_len_logprior`, foldé dans `lp_edge`/`ls_edge` (les deux branches du forward-backward) | `b50_ps2_elo_entbin5_iso_estruct`, `_estruct_ehi` (edge 0.05 compensé), `_estruct_efit` (edge fitté) |
| **C2** — érosion | `smm_scorer.erode_scores()` (min-filter) + `SmmParams.erode_radius`, appliqué en sortie de `score_document` | `b50_ps2_elo_entbin5_iso_er1`, `_er2`, `_er3` |
| **D** — synthétique | `gen_synth.py` (génération 7B + 4 schémas via clés/vendor exacts, labels loggés à la génération, KGW mask + entropie recalculés avec la sémantique production doc-only) + `run_synth.sh` (SLURM) + `validate_synth.py` (KS-test réalisme, verdict imprimé) + `fit_params(synth_docs=...)` (pools d'émission seulement, priors sur docs réels) + `cv_smm.load_synth()` | `b50_ps2_elo_entbin5_iso_synth`, `_synth_kgwxens` |
| **Ensemble** | `ensemble_smm.py` : `--cv cfg1 cfg2 ...` (protocole de folds identique, moyenne de log-odds par doc) et `--combine sub1.jsonl sub2.jsonl` | — |
| **2 seeds** | `cv_smm.py --seed N` (plus besoin d'éditer `SEED`) | — |

**Validation faite ici (sans dataset GPU) :** `python test_roadmap_units.py` → 20/20 OK (érosion == brute-force ; prior de bord nul == legacy exact ; prior structuré = distribution propre non-croissante ; alias `kgw_x` lit bien le signal kgw et se désactive proprement sans lpg ; pooling synth ne touche pas les priors ; round-trip logit). `test_no_adjacent.py` → 5/5 OK (forward-backward intact).

### Séquence de lancement (poste Alexandre ou JURECA — nécessite `output/*.npz`)

```bash
cd cispa_final/task_1_text_watermark/alexandre

# 0. (JURECA, tout de suite, en parallèle de tout le reste) générer le synthétique
sbatch run_synth.sh          # ~2-4 h → output/synth.jsonl + {kgw,entropy}_synth.npz

# 1. Workstream A (10 min/config, CPU) — le levier n°1
python cv_smm.py --configs b50_ps2_elo_entbin5_iso b50_ps2_elo_entbin5_iso_kgwx \
    b50_ps2_elo_entbin5_iso_kgwxens b50_ps2_elo_entbin5_iso_kgwx_c4 \
    b50_ps2_elo_entbin5_iso_kgwxens_c4 b50_ps2_elo_entbin5_iso_unix \
    b50_ps2_elo_entbin5_iso_bothx | tee cv_wsA_seed0.log

# 2. Workstream B+C pendant ce temps (autre terminal)
python cv_smm.py --configs b50_ps2_elo_entbin8_iso b40_ps2_elo_entbin10_iso \
    b50_ps2_elo_entbin6_iso b50_ps2_elo_entbin5_iso_c6 b50_ps2_elo_entbin5_iso_az25 \
    b50_ps2_elo_entbin5_iso_az35 b50_ps2_elo_entbin5_iso_msf10 \
    b50_ps2_elo_entbin5_iso_pse05 b50_ps2_elo_entbin5_iso_pse2 | tee cv_wsB_seed0.log
python cv_smm.py --configs b50_ps2_elo_entbin5_iso_estruct \
    b50_ps2_elo_entbin5_iso_estruct_ehi b50_ps2_elo_entbin5_iso_estruct_efit \
    b50_ps2_elo_entbin5_iso_er1 b50_ps2_elo_entbin5_iso_er2 \
    b50_ps2_elo_entbin5_iso_er3 | tee cv_wsC_seed0.log

# 3. Tout gagnant (> 0.4301) : confirmer sur le second découpage de folds
python cv_smm.py --seed 1 --configs <gagnants> | tee cv_winners_seed1.log

# 4. Ensemble des 2-4 meilleurs (validé en CV d'abord)
python ensemble_smm.py --cv b50_ps2_elo_entbin5_iso <gagnant_A> <gagnant_BC> 
python ensemble_smm.py --cv --seed 1 ...   # confirmer

# 5. Soumission (config simple gagnante OU ensemble via --combine des finals)
python cv_smm.py --final <winner> --out submission_<winner>.jsonl
python ../../shared/submit.py submission_<winner>.jsonl \
  --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "<description precise>"

# 6. Quand le job synth est fini : VALIDER LE RÉALISME AVANT TOUT USAGE
python validate_synth.py     # verdict KS imprimé ; si MISMATCH → régénérer :
# sbatch --export=ALL,TEMPERATURE=0.8,TOP_P=0.95 run_synth.sh
python cv_smm.py --configs b50_ps2_elo_entbin5_iso_synth b50_ps2_elo_entbin5_iso_synth_kgwxens
```

**Rappels de discipline :** décision uniquement sur CV ×2 seeds ; jamais de synthétique dans l'éval (le code l'interdit déjà — pools de fit seulement) ; re-vérifier le format avant soumission (1320 docs, scores ∈ [0,1]) ; logger chaque soumission avec `--method`.
