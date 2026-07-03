# Tâche 1 — Attempt 1 (Alexandre)

**Responsable :** Alexandre (ansart1)
**Code :** [`task_1_text_watermark/alexandre/`](../../task_1_text_watermark/alexandre/)
**Spec :** [`docs/subject/task_1.md`](../subject/task_1.md)

---

## État actuel (2026-07-03, matin)

| Élément | Statut |
| -------- | ------ |
| **Soumission #994 (entropie 7B + isotonic, baseline prod)** | **score public 0.349 — best actuel** |
| Soumission #1513 (idem + multiscale geometric z, w=0.2) | score public **0.3364** — **négatif** vs #994 (§20.3) |
| Calibration T/top-p exact LLR (grille 9 points, `calib_pass.py`) | ❌ **négatif** — meilleur = T=1/p=1 (= raw), exact dans le pipeline dégrade encore le CV (§20.1) |
| Multiscale geometric cover (`multiscale.py`, Point 3) | ❌ **négatif sur public** malgré léger gain CV (+0.3 pt) — rejeté (§20.3) |
| Génération synthétique (`gen_synthetic.py`, Point 2) | ⚙️ code + intégration fit prêts ; job GPU JURECA **en attente** (SSH/TOTP) (§20.2) |

**Pipeline de production inchangé :** `cv_smm.py --final b50_ps2_elo_entbin5_iso` (#994, public 0.349).

---

## État actuel (2026-07-02, nuit)

| Élément | Statut |
| -------- | ------ |
| Pipeline HMM (v1, abandonné) | ❌ remplacé — boundary bleed, score plafonné |
| Pipeline semi-Markov (v2, **retenu**) | ✅ implémenté, testé, soumis |
| Soumission #30 (HMM, sans KGW) | score public **0.0065** |
| Soumission #101 (HMM + KGW) | score public **0.0071** |
| Soumission #159 (semi-Markov, sans KGW) | score public **0.2001** — ×28 vs #101 |
| Soumission #262 (semi-Markov + KGW) | score public **0.2029** |
| Soumission #408 (semi-Markov + KGW + Unigram) | score public **0.2037** |
| Soumission #509 (CV-binned : SMM + LLR empiriques + priors fittés) | score public **0.2526** (+24 % rel. vs #408) |
| Soumission #771 (LLR conditionnés par entropie, proxy 0.5B) | score public **0.328** (+30 % rel. vs #509) |
| **Soumission #994 (idem + entropie 7B exacte + lissage isotonic)** | **score public 0.349** — **best actuel** (+6.3 % rel. vs #771) |
| Val TPR@0.1%FPR — semi-Markov v2 (+ KGW + Unigram) | **0.3137** (neutre vs +KGW seul) |
| KGW | ✅ intégré via `.npz` précalculés (CUDA Philox, JURECA) |
| Unigram | ✅ intégré (`perm@152064`, éligibilité `id < 151643`) — gain marginal, plafond atteint (§9) |
| CV 5-fold + LLR binned (`cv_smm.py`, `fit_smm.py`) | ✅ fait et soumis — CV 0.3484, public 0.2526 (§11) |
| Diagnostic shift CV→public + robustesse prior | ✅ fait — pas de shift de longueurs, prior fitté conservé (§12) |
| Entropy (LLR conditionnés, proxy Qwen 0.5B) | ✅ fait et soumis — CV 0.3685, public 0.328 (§16) |
| **Entropie 7B exacte + lissage isotonic** | ✅ fait et soumis — **CV 0.4301** (+16.7 % rel. vs proxy 0.5B), public **0.349** (§18) |
| Vraisemblance exacte Gumbel-Max/TextSeal (`logp_target`, 7B) | ❌ **testé, négatif** — CV chute de 0.35→0.28, y compris à poids de mixture réduit (§18, §19.3) |
| KGW/Unigram exact (`kgw_lpg`/`unigram_lpg`, 7B) | ❌ **testé, négatif** — bug de référence H0 trouvé + corrigé (§19.2), mais résultat final toujours sous la baseline |
| Recherche bibliographique (WISER, GCD/AOL ACL2025, TextSeal `localized_detect`) | ✅ fait — aucun algorithme publié supérieur au SMM pour ce cadre (longueurs canoniques connues) (§19.1) |

**Meilleur score leaderboard public : 0.349** (LLR conditionnés par entropie 7B exacte + lissage isotonic). Le gros du gain vient (1) du passage HMM → semi-Markov à longueurs discrètes + fix du bug n-grammes répétés, (2) du remplacement des émissions gaussiennes à shifts fixes par des LLR empiriques binnés + priors fittés, sélectionnés en CV 5-fold, (3) du conditionnement des LLR par bin d'entropie prédictive du LM, (4) du passage de l'entropie proxy (Qwen 0.5B) à l'entropie exacte du générateur (Qwen 7B, §18) — le plus gros gain isolé mesuré à ce jour (+16.7 % CV). KGW (+1.4 % relatif) et Unigram (+0.4 % relatif) ajoutent chacun un gain modeste mais net.

**Pipeline de production actuel :** `cv_smm.py --final b50_ps2_elo_entbin5_iso` (nécessite `output/kgw_*.npz` + `output/entropy_*.npz`, tous deux désormais calculés avec le 7B — voir §18). Le pipeline gaussien `run_smm.py` (reproduit #408) reste disponible en fallback sans fichiers d'entropie.

---

## 1. Objectif et métrique

Pour **chaque token** de chaque document test, prédire un score ∈ [0, 1] : confiance que le token a été généré **pendant qu'un watermarking était actif** (label binaire 0/1 en train/val, caché en test).

La métrique est **TPR @ 0.1% FPR**, calculée sur **tous les tokens poolés** (pas par document) : on fixe le seuil τ tel que ≤ 0.1 % des tokens clean (label 0) aient un score ≥ τ, puis on mesure la fraction de tokens watermarkés au-dessus de ce même τ. Budget FPR minuscule sur val (51 tokens clean sur 90 docs) — les quelques tokens clean les mieux notés déterminent tout.

**Quatre familles** de watermark possibles, souvent mélangées : TextSeal, Gumbel-Max, Unigram, KGW. On ne prédit pas *quel* schéma — seulement *watermark actif ou non*.

---

## 2. Pourquoi le HMM (v1) a été abandonné

Le premier pipeline (HMM à états `clean` / `schéma i`, forward-backward) donnait val 0.0185. Analyse post-#101 :

- **Boundary bleed** : le posterior se répand sur les tokens clean adjacents. Sur les 51 FPs à 0.1 % FPR, 31 étaient à < 10 tokens d'une frontière de span réelle.
- **KGW dégradait le val** (0.0185 → 0.0066) — problème structurel, pas de calibration.
- **Cause racine** : bug de signal sur n-grammes répétés (§4), pas seulement le modèle.

---

## 3. Infrastructure

| Ressource | Chemin |
| --------- | ------ |
| Dataset (local) | `cispa_final/data/watermark_localization/{train,validation,test}.jsonl` |
| Dataset (cluster) | `/p/scratch/training2625/ansart1/loki/watermark_localization/` |
| Clés + params détecteurs | [`task_1_text_watermark/watermark_config.yaml`](../../task_1_text_watermark/watermark_config.yaml) |
| Repos vendor (commits épinglés) | `task_1_text_watermark/vendor/{textseal,lm-watermarking,unigram-watermark}/` |
| API | `CISPA_BASE_URL` + `CISPA_API_KEY` · task id `30-watermark-localization` |
| Soumission / historique | `shared/submit.py` → `history/submissions.jsonl` |
| Eval locale + viewer | `shared/task1_eval.py` → `history/task1_viz/*.json` |

**Tokenizer :** `Qwen/Qwen2.5-7B-Instruct`. Utiliser les `token_ids` du JSONL — pas de retokenization du champ `text`.

**Volumes :** train 90 · val 90 · test 1 320 docs (~1,4 M tokens).

---

## 4. Bug trouvé : évidence fantôme sur n-grammes répétés

Les PRF (TextSeal, Gumbel-Max, KGW) seedent leur tirage sur un n-gramme de contexte. **Si le même n-gramme apparaît deux fois, le PRF ré-émet le même tirage** — pas une observation indépendante.

Un doc val contenait `letters . letters . ...` (40+ fois) dans une zone **clean**, z-score fenêtre à **34** — absorbait la moitié du budget FPR.

**Fix (`detectors.py::_dedup_mask`)** : seule la **première occurrence** de chaque n-gramme porte le signal ; répétitions → H0 neutre. Comportement correct d'un détecteur, pas heuristique ad hoc.

---

## 5. Signaux par token (`detectors.py` + `smm_scorer.py`)

| Schéma | Signal | Clés / params (YAML) | H0 |
| ------ | ------ | -------------------- | -- |
| **TextSeal** | Dual-key Gumbel : `α·g(r_a)+(1-α)·g(r_b)`, `g(r)=-log(1-r)` | `key_a`, `key_b`, `ngram=3`, `α=0.5` | μ=1, σ²=0.5 |
| **Gumbel-Max** | `-log(1-r)` sur PRF uniforme | `secret_key`, `ngram=2` | μ=1, σ²=1 |
| **Unigram** | 1 si token ∈ greenlist fixe | `watermark_key`, `fraction=0.5`, **perm@152064** | Bernoulli(0.5) |
| **KGW** | binaire green/red, précalculé GPU | `gamma=0.25`, schéma `ff-anchored_minhash_prf-4-True-1306382177` | Bernoulli(0.25) |

**PRF CPU :** `vendor/textseal/.../core.py` (`prf_uniform`) via `importlib` — évite la dépendance `nltk`.

**Unigram :** intégré dans `smm_scorer.py::_unigram_signal`. Dédup par **token unique** (greenlist context-free). Éligibilité : `token_id < 151643` (tokens spéciaux exclus, confirmé organisateur). Permutation : `152064` (contrainte génération — voir §9).

**KGW :** masques dans `output/kgw_{train,validation,test}.npz`, clés = `document_id` exacts du JSONL. Même dédup n-grammes que TextSeal/Gumbel (contexte 3 tokens).

---

## 6. Modèle retenu : semi-Markov à segments (`smm_scorer.py`)

### Pourquoi semi-Markov

Longueurs de spans quasi discrètes (train+val) :

```
31: 64   47: 133   63: 163   95: 145   159: 158   320: 73   (93 %, reste = longue traîne)
```

Un HMM token-à-token ne peut pas exploiter ça — il "devine" les frontières une transition à la fois (bleeding v1). Le semi-Markov score des fenêtres de longueur candidate et fait forward-backward sur les *segmentations*.

### Structure

- Document = segments : token clean isolé (coût nul) ou span watermarkée `L ∈ LENGTHS` avec prior empirique `p(L)` (+ 7 % masse résiduelle hors-liste / bords tronqués).
- **Émission** (3 modes, sélectionnables via `SmmParams`) :
  - `"gaussian"` : décalage gaussien sur z-scores, `logsumexp` sur 3 shifts par schéma — **defaults = pipeline #408**
  - `"binned"` : LLR empirique par bins, fitté sur train (`fit_smm.py`) — **en test via CV**
  - `"bernoulli"` : LLR fermé pour KGW (binaire)
- **Spans tronquées** : hypothèses préfixe/suffixe avec prior dédié.
- **Score token** : posterior exact `P(token ∈ span wm | doc)`, log-odds + sigmoïde température 40.

### Résultats

| Version | Val TPR@0.1%FPR | Score public |
| ------- | ---------------- | ------------ |
| HMM v1 (sans KGW) | 0.0140 | 0.0065 (#30) |
| HMM v1 + KGW | 0.0066 | 0.0071 (#101) |
| Semi-Markov v2 (sans KGW) | 0.2822 | 0.2001 (#159) |
| Semi-Markov v2 + KGW | 0.3135 | 0.2029 (#262) |
| **Semi-Markov v2 + KGW + Unigram** | **0.3137** | **0.2037 (#408)** |

---

## 7. Approches intermédiaires testées et abandonnées

| Approche | Fichier | Val TPR@0.1%FPR | Pourquoi abandonnée |
| -------- | ------- | ---------------- | ------------------- |
| Fusion z-score + sigmoïde | `build_scores.py` | ~0.0016 | Pas de structure de span |
| Régression logistique | `train_calibrator.py` | ~0.0070 | Token-à-token, pas de segments |
| HMM forward-backward | `hmm_scorer.py` | 0.0185 | Boundary bleed (§2) |
| Matched filter + `min(g,d)` | `matched_filter.py` | ~0.023 | Glouton, perte d'info vs posterior |

Conservées en référence — **ne pas réutiliser pour une soumission**.

---

## 8. Intégration KGW

KGW nécessite `torch.randperm` sur CUDA (Philox) — impossible à recalculer sur CPU (~75 % des tokens KGW ressortent faussement clean). Masques précalculés sur JURECA (`kgw_scores.py`, job `15399747`, ~6 min A100) → `output/kgw_{train,validation,test}.npz`.

**Branchement :** `run_smm.py --kgw output/kgw_test.npz`

| | Val TPR@0.1%FPR | Score public |
| -- | --------------- | ------------ |
| Sans KGW | 0.2822 | 0.2001 (#159) |
| Avec KGW | 0.3135 (+11 % rel.) | 0.2029 (#262, +1.4 % rel.) |

**Gain val >> gain test** (+11 % vs +1.4 %) — attendu (§10). Hypothèse `vocab_size` KGW **écartée** : probe `--auto` a choisi `151665` (z 9.0–9.3 sur 4 candidats, quasi équivalents). Causes restantes non vérifiées : mix de schémas test ≠ val ; shifts KGW sweepés sur val.

---

## 9. Unigram — config, détection, plafond

### Config finale (ne plus toucher)

| Paramètre | Valeur | Justification |
| --------- | ------ | ------------- |
| Permutation (`MASK_VOCAB`) | **152064** | Contrainte génération : `GPTWatermarkLogitsWarper` additionne le masque aux logits → dimension = `config.vocab_size` du modèle Qwen |
| Éligibilité (`BASE_VOCAB`) | **151643** | Organisateur (Maitri Shah) : tokens spéciaux hors greenlist ; `token_id >= 151643` → signal neutre |
| Dédup | token unique | Même logique que vendor `.unidetect()` — répétitions n'ajoutent pas d'évidence |
| Shift H1 | `(0.5,)` seul | Signal faible (`strength=1.0`) et rare — grille large = risque FPR |

Code : `unigram_scan.py` (scan + greenlist), `smm_scorer.py::_unigram_signal` (scoring).

### Comment le signal a été trouvé

Après deux fausses pistes (mauvais `vocab_size` + comptage avec répétitions = même bug que §4), le scan non supervisé `unigram_scan.py` a fonctionné :

1. **Contraindre `vocab_size` par la physique de génération** (152064, pas scan aveugle).
2. **Tester sur test set non labellé** (1 320 docs) avec **null empirique** (20–50 clés leurres) plutôt que moyennes sur petits pools dilués.
3. Statistique : max z-score de fraction verte (tokens uniques) par fenêtre glissante aux longueurs canoniques.

Résultat test (vraie clé vs 50 leurres, `vocab=152064`) : rang **1/51** sur docs avec fenêtre z>4 (28 vs moyenne 5.7) et masse top-20 (86.2 vs 78.4). Vérification empirique des candidats `vocab_size` : `151643` se comporte comme une clé aléatoire (rang 21/21), `152064` rang 1/21.

### Impact mesuré — plafond atteint

| Métrique | Valeur |
| -------- | ------ |
| Val TPR@0.1%FPR | 0.3137 (neutre vs 0.3135 sans Unigram) |
| Score public #408 | **0.2037** (+0.4 % rel. vs 0.2029) |

Gain val nul — cohérent : ~10–20 spans Unigram sur 396 dans val, signal trop faible et rare pour bouger un TPR global.

**Contre-vérification rigoureuse (null 20 clés leurres, pool résiduel + test set) :** le signal Unigram n'est **pas diffus** — il tient essentiellement à **un document test** (`doc_id=748`, z=5.37, nettement au-dessus du max leurre 4.95). Le 2ᵉ candidat (`doc_id=91`, z=4.52) est dans la plage des leurres. Aucune statistique (moyenne, queue, fenêtres non-canoniques) ne sort du bruit sur le pool résiduel labellé.

**Verdict pour les agents : Unigram est intégré et correctement calibré. Ne pas investir plus de temps dessus** — plafond physique, pas méthodologique. Continuer à chercher du signal Unigram supplémentaire n'est pas un bon ROI.

**Leçon retenue :** ne pas conclure "pas de signal" sur des tests de moyenne sous-puissants ; contraindre les hyperparams par la génération + null empirique sur toutes les données disponibles.

**Bug de reproductibilité trouvé et corrigé (2026-07-02, 19h) :** avant ce fix, `smm_scorer.py::doc_signals()` calculait le signal Unigram (dans `detectors.py`) mais ne le remontait jamais — et `DEFAULT_SHIFTS` n'avait pas d'entrée `unigram`. Résultat : relancer `run_smm.py` (pipeline gaussien, non fitté) produisait un score **sans** Unigram, contrairement à ce que ce document affirmait déjà. La soumission `#408` a donc probablement été générée par un script ponctuel jamais committé. **Corrigé** : `doc_signals()` calcule et retourne maintenant nativement le signal Unigram (`_unigram_signal`, perm@152064, éligibilité <151643, dédup token unique), et `DEFAULT_SHIFTS["unigram"] = (0.5,)`. Vérifié : val locale 0.3142 (cohérent avec 0.3137 attendu), test régénéré quasi identique à l'ancien fichier `#408` (1320/1320 docs, écart moyen 9×10⁻⁶) — pas re-soumis (équivalence déjà validée localement, cooldown API préservé).

**Portée de ce fix :** il ne concerne que le pipeline gaussien par défaut (`run_smm.py`, non fitté, reproduit `#408`) — **pas** le pipeline CV-binned retenu en §11 (`cv_smm.py`), qui reste volontairement sans Unigram (`include_unigram=False` par défaut dans `fit_smm.py::fit_params`, testé explicitement et dégradant le score, cf. §11). Les deux constats coexistent et ne se contredisent pas : sur le pipeline simple, Unigram apporte un micro-gain net (+0.4 %) ; une fois les autres signaux mieux calibrés (LLR empiriques), le même Unigram devient net-négatif — cohérent avec un signal si rare qu'il ne vaut la peine que si le reste du budget FPR n'est pas déjà optimisé ailleurs.

---

## 10. Le score val transfère-t-il tel quel au leaderboard ?

**Non.** Raisons :

- **Optimisme val attendu** : debug des FPs sur val (§4), shifts KGW sweepés sur val (§8). Bugfix légitime, mais affinage observé sur val.
- **Budget FPR bruyant** : val = 51 tokens clean ; test = ~15× plus de docs.
- **Écarts observés** : sans KGW, 0.2822 → 0.2001 (−30 % rel.). Avec KGW, +11 % val → +1.4 % public. L'ordre de grandeur ×28 vs HMM reste le signal principal.
- Score public = sous-ensemble visible ; **subset caché** détermine le classement final (mention API).

**Métrique de décision pour les prochains changements :** TPR@0.1%FPR en **CV 5-fold document-level** sur train+val (`cv_smm.py`) — plus robuste que val seule.

---

## 11. CV 5-fold + LLR empiriques binnés → nouveau best public 0.2526

### Pourquoi : deux problèmes du pipeline #408

1. **Émissions gaussiennes à shifts fixes** : la vraie distribution H1 des signaux n'est pas un simple décalage gaussien, et les shifts avaient été réglés à la main (partiellement sur val seule).
2. **Sélection de modèle sur val seule** = 90 docs, budget FPR de 51 tokens — trop bruyant pour piloter des choix fins (§10).

### Test préalable — l'hypothèse "shifts faibles" ne marche pas (résultat négatif utile)

Avant le CV, l'hypothèse "capturer les spans à signal faible en ajoutant des hypothèses de shift plus basses" (motivée par le pool résiduel à z moyen ≈ 0.75) a été testée proprement sur train **et** val (`sweep_shifts.py`, `sweep_prior.py`) :

| Config | Train | Val |
| ------ | ----- | --- |
| Baseline (shifts #408, `p_span=0.004`) | 0.2518 | **0.3140** |
| + shift KGW faible (0.3) | 0.2587 | 0.2900 |
| + shifts faibles partout (TS/GM/KGW) | 0.2600 | 0.2897 |
| Grille dense 5 shifts/schéma | 0.2471 | 0.3101 |
| `p_span=0.002` / `0.008` / `0.016` | 0.2573 / 0.2498 / 0.2347 | 0.3114 / 0.3033 / 0.2665 |

**Verdict : diluer les hypothèses fortes avec des shifts faibles coûte plus en FPR (tokens clean remontés) que ça ne rapporte en TPR.** Les spans faibles ne sont pas capturables par ce levier ; les priors par défaut étaient déjà quasi optimaux. C'est ce qui a motivé le passage aux LLR empiriques (apprendre la forme réelle de H1 au lieu d'élargir une famille paramétrique).

### Méthode retenue (`fit_smm.py` + `cv_smm.py`)

- **Émissions binnées** : LLR empirique par bins de quantiles H0, fitté sur les pools étiquetés (spans assignées par schéma via z de fenêtre ≥ 3 ; KGW en Bernoulli fermé). Remplace la grille de shifts gaussiens.
- **Priors fittés** : longueur de spans, `p_span`, prior de bord estimés par comptage sur les docs de fit.
- **Sélection en CV 5-fold document-level** sur train+val (180 docs) : chaque doc est scoré par un modèle fitté sans lui ; TPR@0.1%FPR poolé sur les 180 docs held-out. Plus robuste que val seule.

### Résultats CV (extraits de la grille, 19 configs)

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| baseline (gaussien #408, non fitté) | 0.2764 |
| priors fittés, émissions gaussiennes | 0.2743 (neutre) |
| binned 30 bins | 0.2971 |
| binned 30 + edge prior 0.005 | 0.3202 |
| binned 40 + edge 0.005 + p_span ×2 | 0.3464 |
| **binned 50 + edge 0.005 + p_span ×2 (retenu)** | **0.3484** (0.3516 seed 1) |
| retenu + émission Unigram rebranchée | 0.3408 ⚠️ dégrade |

Le gain vient surtout des **émissions binnées** (+0.02) et du **prior de bord abaissé** (+0.02) — les priors fittés seuls n'apportent rien, c'est la combinaison LLR empirique + edge bas qui paie.

**Note Unigram :** la config retenue n'inclut **pas** l'émission Unigram. Testé explicitement (`binned50_edge_lo_ps2_uni`) : rebrancher Unigram dans le pipeline CV-binned fait passer le CV de 0.3484 à 0.3408. Cohérent avec le plafond de signal (§9) : ~1 doc test réellement attribuable, contre un coût FPR diffus sur tous les autres. Le pipeline #408 (gaussien + Unigram) reste reproductible via `run_smm.py`, mais le best actuel (#509) est sans Unigram.

### Soumission

Refit sur les 180 docs, scoring test (`cv_smm.py --final binned50_edge_lo_ps2`), soumis → **#509, public 0.2526** (+24 % relatif vs 0.2037). Le gain CV (+26 % relatif) transfère presque entièrement au public — contrairement aux réglages sur val seule (§10), ce qui valide le CV comme métrique de décision.

**Vérifications faites avant soumission :** refactor de `smm_scorer.py` validé par test de régression (scores identiques à #262 à < 1e-12) ; config gagnante revalidée avec un second découpage de folds (seed 1 → 0.3516, stable) ; format de soumission validé (1320 docs, scores ∈ [0,1]).

---

## 12. Diagnostic du shift CV→public + robustesse du prior de longueur

### Écart CV→public : distribution, pas overfitting

Ratio public/CV **constant** (~73 %) avant et après le tuning CV : baseline 0.2764→0.2029 (73.4 %), winner #509 0.3484→0.2526 (72.5 %). Les gains transfèrent à ~100 % — nos décisions généralisent ; l'écart de *niveau* vient de la distribution du test.

### Diagnostic sans labels (`diagnose_shift.py`)

Comparaison labeled held-out (CV winner, 5-fold) vs test (`submission_smm_cv.jsonl`) :

| Statistique | Labeled | Test |
| ----------- | ------- | ---- |
| Quantiles de scores (p50/p90/p99/p99.9) | 0.4960 / 0.5357 / 0.5636 / 0.5794 | 0.4968 / 0.5362 / 0.5642 / 0.5808 |
| Fraction tokens ≥ τ@0.1%FPR | 0.1635 | 0.1714 |
| Runs canoniques ±4 (seuil strict) | 27.6 % | 28.1 % |
| Longueurs non-canoniques dominantes | 36–42, 55–57, 81–89 | 36–42, 53–57, 86–87 |
| Docs avec ≥ 1 détection | 161/180 | 1205/1320 |

**Verdict : aucune évidence de longueurs de spans inconnues au test** — les histogrammes de run-lengths sont superposés, les longueurs "non-canoniques" sont les mêmes résidus de bordure des deux côtés, et seuls 6 runs test dépassent 324 (max 568, compatible avec deux spans adjacentes fusionnées). L'écart de niveau CV→public n'est **pas actionnable** par le prior de longueur ; il reflète vraisemblablement la difficulté intrinsèque des textes test et le pool clean 15× plus grand.

### Robustesse du prior de longueur (CV 5-fold)

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| **winner (prior fitté, #509)** | **0.3484** |
| mélange uniforme m=0.1 | 0.3452 |
| mélange uniforme m=0.2 | 0.3389 |
| m=0.1 + plage étendue (20, 401) | 0.3384 |
| prior uniforme pur | 0.2790 |

Le prior uniforme pur s'effondre (−7 pts) : **le prior fitté porte une vraie information**, ce n'est pas un artefact. Toute assurance uniforme coûte du CV, et le diagnostic ci-dessus ne montre aucun risque test qui la justifierait. **Décision : prior fitté conservé tel quel, pas de soumission A/B** (budget préservé). Implémentation : `fit_length_prior(mix_uniform=, len_range=)` dans `fit_smm.py`, configs `b50_ps2_elo_{mix10,mix20,mix10_ext,unif}` dans `cv_smm.py`.

---

## 13. Prochaines étapes, par impact attendu

1. ~~Entropies 7B exactes~~ — **fait, gros gain confirmé** : CV 0.3685→0.4301 (§18).
2. **Couverture géométrique** (`localized_detect` vendor TextSeal) — spans diluées dans docs mixtes. Reste à faire.
3. **Raffiner la grille CV** autour de `b50_ps2_elo_entbin5_iso` (ent bins, clip, edge) avec les nouveaux signaux 7B — rendement décroissant attendu mais pas testé.
4. ~~Vraisemblance exacte Gumbel-Max/TextSeal (`logp_target`)~~ — **testée, négative** (§18).
5. **`unigram_lpg` / `kgw_lpg`** (probabilité verte "boostée" par logits, calculée par `entropy_pass.py` mais jamais câblée dans `smm_scorer.py`) — signal produit mais non exploité ; piste non testée, priorité basse vu le plafond déjà documenté sur Unigram (§9) et le score déjà correctement calibré pour KGW.
6. ~~Pondération multiplicative par entropie~~ — **testée, négative** ; la version conditionnée (LLR par bin d'entropie) est celle qui paie (§16).
7. ~~Prior de longueur (mixing/uniforme)~~ — **testé, négatif** (§12).
8. ~~Shifts faibles / priors à la main~~ — **testé, négatif** (§11).
9. ~~Unigram~~ — **résolu, plafond atteint** (§9).
10. ~~Hypothèse vocab KGW~~ — **écartée** (§8).

---

## 14. Comment exécuter

### Pipeline best actuel (public 0.349) — LLR conditionnés par entropie 7B + isotonic

Prérequis : `output/kgw_*.npz` (§8) **et** `output/{entropy,logp,unigram_lpg,kgw_lpg}_*.npz` (7B, §18 — seul `entropy_*` est utilisé par ce config).

```bash
cd cispa_final/task_1_text_watermark/alexandre

python cv_smm.py --configs b50_ps2_elo_entbin5_iso              # revalider en CV (0.4301)
python cv_smm.py --final b50_ps2_elo_entbin5_iso --out submission_entbin5_iso_7b.jsonl
python ../../shared/submit.py submission_entbin5_iso_7b.jsonl \
  --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "SMM + entropy-conditioned binned LLR (7B exact entropy) + isotonic smoothing"
```

### Précalcul signaux GPU 7B (si `output/entropy_*.npz` etc. absents)

```bash
# Sur JURECA, depuis le login node (nécessite le 7B en cache HF_HOME, ~15 GB)
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre
sbatch run_entropy.sh
# -> output/{entropy,logp,unigram_lpg,kgw_lpg}_{train,validation,test}.npz (~14 min A100)
# Rapatriement (tar pour éviter les soucis de wildcard scp) :
#   ssh jureca 'cd code/cispa_final/task_1_text_watermark/alexandre/output && tar czf /tmp/gpu_signals.tgz entropy_*.npz logp_*.npz unigram_lpg_*.npz kgw_lpg_*.npz'
#   scp jureca:/tmp/gpu_signals.tgz output/ && tar xzf output/gpu_signals.tgz -C output/
```

### Pipeline #509 (public 0.2526) — CV-binned sans entropie

```bash
python cv_smm.py --final binned50_edge_lo_ps2 --out submission_cv.jsonl
```

### Pipeline #408 — semi-Markov + KGW + Unigram (defaults non fittés)

Prérequis : dataset dans `data/watermark_localization/`, masques KGW dans `output/kgw_*.npz`.

```bash
cd cispa_final/task_1_text_watermark/alexandre

# Val locale
python run_smm.py --split validation --out val_scores.jsonl --kgw output/kgw_validation.npz
python ../../shared/task1_eval.py \
    --dataset ../../data/watermark_localization/validation.jsonl \
  --predictions val_scores.jsonl \
  --method "semi-Markov + KGW + Unigram"

# Soumission test
python run_smm.py --split test --out submission.jsonl --kgw output/kgw_test.npz
python ../../shared/submit.py submission.jsonl \
    --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "semi-Markov + KGW + Unigram"
```

### Précalcul KGW (si `.npz` absents)

```bash
# Sur JURECA
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre
sbatch run_kgw.sh   # -> output/kgw_{train,validation,test}.npz

# Rapatriement local
bash cispa_final/scripts/fetch_kgw.sh
```

---

## 15. Carte des fichiers

| Fichier | Rôle |
| ------- | ---- |
| `detectors.py` | Signaux PRF CPU (TextSeal, Gumbel-Max) + dédup n-grammes |
| `unigram_scan.py` | Greenlist Unigram (perm 152064) + scan non supervisé |
| `smm_scorer.py` | **Modèle retenu** : semi-Markov, émissions gaussian/binned/bernoulli + variantes `*_ent` conditionnées par entropie |
| `run_smm.py` | CLI scoring → JSONL, `--kgw` pour KGW (pipeline gaussien #408) |
| `fit_smm.py` | Fit supervisé priors + LLR binned (± conditionnés entropie) |
| `cv_smm.py` | CV 5-fold document-level + `--final` pour soumission |
| `diagnose_shift.py` | Diagnostic shift labeled vs test sans labels (§12) |
| `audit_results.py` | Audit local CV : FP/FN, breakdown par scheme, edge vs middle (§17) |
| `test_no_adjacent.py` | Validation brute-force du fix boundary-bleed (§17) |
| `kgw_scores.py` | Masques KGW (CUDA Philox, JURECA) |
| `run_kgw.sh` | SLURM précalcul KGW |
| `entropy_pass.py` | Pass GPU combiné (7B) : entropie, `logp_target` (LLR exact), `unigram_lpg`, `kgw_lpg` |
| `run_entropy.sh` | SLURM précalcul du pass GPU combiné, 7B (§18) |
| `remote_setup_entropy.sh` | Setup login node (deps, cache modèle) + sbatch *(historique, proxy 0.5B)* |
| `download_entropy_model.py` | Pré-téléchargement modèle proxy (login node) *(historique)* |
| `poll_entropy.sh` | Poll WSL du job entropie |
| `../../scripts/fetch_kgw.sh` | Rapatriement `.npz` depuis JURECA |
| `sweep_shifts.py`, `sweep_prior.py` | Sweeps shifts/priors (train+val) |
| `hmm_scorer.py`, `run_hmm.py` | *(abandonné v1)* |
| `matched_filter.py`, `build_scores.py`, `train_calibrator.py` | *(abandonnés)* |

**Sorties soumises :** `submission_entbin5_iso_7b.jsonl` (**#994, 0.349, best**), `submission_smm_entbin.jsonl` (#771, 0.328), `submission_smm_cv.jsonl` (#509, 0.2526), `submission_smm_kgw_uni.jsonl` (#408, 0.2037), `submission_smm_kgw.jsonl` (#262, 0.2029), `submission_smm_nokgw.jsonl` (#159, 0.2001). `submission_smm_cvbinned.jsonl` = même config `binned50_edge_lo_ps2` regénérée indépendamment (non soumise, redondante).

---

## 16. Entropy — LLR conditionnés par bin d'entropie → best public 0.328

### Théorie (TextSeal §3.2)

Un token à faible entropie est quasi déterministe pour le LM : le PRF du watermark n'a eu aucune influence sur son choix → aucun signal, mais du bruit qui consomme du budget FPR. À l'inverse, les tokens à haute entropie portent tout le signal.

### Données : entropies prédictives par token

`entropy_pass.py` (JURECA, 1 GPU, job `15401286`) : forward pass `Qwen2.5-0.5B-Instruct` (proxy — le 7B n'était pas en cache cluster ; TextSeal Fig. 6 valide le proxy) sur les 3 splits → `output/entropy_{split}.npz` (float16, ~40 s pour 1 500 docs). Position 0 = sentinelle −1 (pas de contexte).

### Deux variantes testées — seule la version *apprise* marche

**1. Pondération multiplicative (plan initial) : négatif.** Poids `w = 0.1 + 0.9·f(Ĥ)` (f = linéaire ou √, normalisation percentile 5–95 fittée sur folds de fit) multipliant les LLR : CV 0.3027 (linéaire) / 0.3388 (√) vs 0.3484 baseline. Explication : sous H0 le LLR a un drift négatif (−KL) ; réduire |LLR| des tokens à basse entropie *remonte* les tokens clean vers 0 et casse la séparation au seuil 0.1 % FPR.

**2. LLR conditionnés par entropie (retenu) : +2 pts CV.** Au lieu d'imposer une forme de poids, on **apprend** des tables LLR séparées par bin d'entropie (bins = quantiles H0) : `fit_binned_llr_ent` (TextSeal/Gumbel : table signal × entropie) et `fit_bernoulli_kgw_ent` (taux de verts H1 par bin d'entropie). Le fit découvre seul que le signal est plus discriminant à haute entropie — et gère correctement le drift H0 par construction (chaque table est un vrai LLR).

| Config | CV seed 0 | CV seed 1 |
| ------ | --------- | --------- |
| binned50 (#509, sans entropie) | 0.3484 | 0.3516 |
| entbin3 × 30 bins | 0.3485 | — |
| entbin3 × 50 bins | 0.3544 | — |
| entbin5 × 30 bins | 0.3652 | — |
| **entbin5 × 50 bins (retenu)** | **0.3685** | **0.3545** |
| entbin7 × 20/30 bins | 0.3536 / 0.3567 | — |

### Soumission

Refit 180 docs + scoring test (`cv_smm.py --final b50_ps2_elo_entbin5`) → **#771, public 0.328** (+30 % rel. vs 0.2526, `improved: true`). Le gain public (+30 %) dépasse même le gain CV (+6 %) — le conditionnement par entropie aide davantage sur la distribution test.

**Levier restant documenté en §13 :** entropies exactes du 7B au lieu du proxy 0.5B.

---

## 17. Audit local + 3 pistes indépendantes (boundary bleed, lissage isotonic, mixture)

En parallèle du travail sur la vraisemblance exacte / KGW par position / données synthétiques (gros leviers GPU, cf. §13), audit **local, sans GPU ni nouvelle donnée** du pipeline #771 (5-fold CV, `audit_results.py`) pour trouver des pistes orthogonales.

### Constats de l'audit

| Constat | Détail |
| ------- | ------ |
| TPR très inégal par scheme | Gumbel-Max 62 % des spans détectés, TextSeal 44 %, **KGW 9 %** (malgré valid_frac=0.94) — confirme que le Bernoulli global KGW est le maillon faible (déjà ciblé par l'agent parallèle) |
| **34 % des spans (269/792) sans signal exploitable**, tous schemes confondus | Score moyen 0.494 (≈ bruit H0 ≈0.497). Cause identifiée : longueur moyenne 73 tokens vs 122–164 pour les spans détectés → pur effet √L (SNR), pas un 5ᵉ scheme caché. Devrait bénéficier de façon disproportionnée à la vraisemblance exacte par token (§13, point 1), qui supprime le bruit d'estimation des tables binnées justement handicapant à petite longueur |
| **Faux positifs = boundary bleed, pas bruit indépendant** | Sur 4 clusters de FP (runs de 6–10 tokens consécutifs) fixant le seuil à 0.1 % FPR, 3/4 sont à 1–5 tokens d'un vrai span (distances 1, 3, 5) |
| Spans en bordure de document sous-détectés | 11 % détectés vs 31 % pour les spans internes (88 vs 704 spans) |
| Longueurs canoniques identiques entre schemes | Piste "prior (scheme, longueur) joint" écartée — rien à gagner |

### Trois fix testés en CV (5-fold, 180 docs)

**A. Interdiction structurelle de deux spans adjacents sans token clean entre eux.** Forward-backward à 2 phases ("après clean" / "après span") ; **losslessy cohérent avec le label truth** (deux positions label=1 consécutives sont toujours fusionnées en un seul span par construction, donc aucune vraie segmentation n'a ce pattern). Validé exact à la précision machine par force brute sur 5 documents synthétiques (`test_no_adjacent.py`). **Gain CV : +0.0006 (bruit)** — le mécanisme dominant des FP n'est donc pas la "double span hallucinée" mais plutôt l'imprécision de frontière d'un span unique (fenêtre glissante mal calée), non couverte par ce fix. **Implémenté (`SmmParams.forbid_adjacent_spans`), non retenu** (complexité non justifiée par le gain).

**B. Lissage isotonic des tables LLR binnées.** Les 4 signaux sont construits pour que "signal plus fort ⇒ plus d'évidence H1" ; imposer une LLR non-décroissante par bin (`IsotonicRegression`, pondérée par comptage) réduit le bruit d'estimation des bins extrêmes — justement ceux qui fixent le seuil à 0.1 % FPR, estimés sur très peu de tokens (180 docs). **Gain CV reproductible : +0.0037 (seed 0, 0.3685→0.3722) et +0.0064 (seed 1, 0.3545→0.3609).** **Retenu.**

**C. Poids de mixture non uniformes par prévalence de scheme.** Le modèle mixe actuellement GM/TS/KGW à poids égal (1/n_hyp) dans le logsumexp ; poids réels ≈ 41/36/22 % parmi les 523 spans assignés avec confiance. **Résultat : négatif partout** (binned: 0.3484→0.3402 ; entbin5: 0.3685→0.3636 ; entbin5+iso: 0.3722→0.3699). Explication probable : sous-pondérer un scheme réduit directement la sensibilité du modèle à ce scheme sans bénéfice structurel équivalent sur le FPR — l'estimation de prévalence (523 spans, bruitée) ne compense pas la perte de sensibilité. **Rejeté.**

### Meilleure config résultante

`b50_ps2_elo_entbin5_iso` (= #771 + lissage isotonic) : **CV 0.3722** vs 0.3685 (#771), soit +1 % relatif. Gain réel mais modeste — **pas encore soumis** : l'agent parallèle modifie activement les mêmes fichiers (`smm_scorer.py`, `fit_smm.py`, émissions `*_exact`) ; le lissage isotonic est une étape générique orthogonale à la source du LLR (empirique binné vs forme fermée) et devrait se composer proprement avec leur travail — préférable de ne pas consommer un slot de soumission sur ce gain isolé avant convergence.

**Fichiers ajoutés :** `audit_results.py` (audit CV : FP/FN, breakdown par scheme, edge vs middle), `test_no_adjacent.py` (validation brute-force du fix A). `fit_binned_llr(..., isotonic=)`, `SmmParams.forbid_adjacent_spans`, `SmmParams.mix_log_weights` dans `smm_scorer.py`/`fit_smm.py`.

---

## 18. Pass GPU 7B combiné : entropie exacte (gros gain) + vraisemblance exacte (négatif) → nouveau best public 0.349

### Objectif

Le proxy Qwen2.5-0.5B (§16) donnait un gain mesuré mais son entropie prédictive **n'est pas celle du vrai générateur** — potentiellement bruitée sur les tokens où 0.5B et 7B divergent. En parallèle, `logp_target` (probabilité que le LM assigne au token réellement généré) permet en théorie de calculer la **vraisemblance exacte** Gumbel-Max/TextSeal sous H1 (détecteur Aaronson : sous watermark, le token choisi maximise `r_v^(1/p_v)`, donc `r | H1 ~ (1/p) r^(1/p-1)` contre `r | H0 ~ Uniform(0,1)` — LLR en forme close, sans passer par des bins empiriques).

Un seul pass GPU (`entropy_pass.py`, réécrit) calcule les quatre signaux en une passe forward sur le **7B** (le générateur réel) : `entropy`, `logp_target`, et par bonus `unigram_lpg`/`kgw_lpg` (probabilité verte "boostée" par les logits, pas encore câblée dans le scorer — cf. §13 point 5).

### Exécution GPU (JURECA)

Job `15401646`, 1×A100, ~14 min, 3 splits (train/val/test, ~1,4 M tokens) — `run_entropy.sh` avec `--require-primary` pour forcer le 7B (pas de fallback silencieux vers un proxy).

**Obstacles résolus en cours de route** (cf. résumé de session) : cache HF du 7B introuvable avec `local_files_only=True` sur le nœud de calcul (fichier `refs/main` manquant après une copie `cp -rL` depuis un cache partagé) ; `HF_HOME` écrasé par `~/.bashrc` dans l'environnement SLURM ; module `unigram_scan.py` absent du dépôt côté cluster (dépôt pas à jour, réuploadé). Aucun de ces problèmes n'affecte la validité des résultats finaux — le job a tourné avec le vrai 7B, vérifié via l'erreur explicite ajoutée dans `load_model()` si `vocab_size != UNIGRAM_VOCAB`.

### Résultat 1 — entropie 7B exacte conditionnée : gros gain confirmé

CV 5-fold, même config (`b50_ps2_elo_entbin5`), seule l'entropie change (proxy 0.5B → 7B) :

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| binned50 sans entropie (#509) | 0.3484 |
| entbin5×50, entropie proxy 0.5B (#771) | 0.3685 |
| **entbin5×50, entropie 7B exacte** | **0.4146** (+12,5 % rel. vs proxy) |
| **+ lissage isotonic (§17.B)** | **0.4301** (+16,7 % rel. vs proxy) |

C'est de loin le plus gros gain isolé mesuré depuis le passage aux LLR binnés (§11) — confirme l'hypothèse de §13 (proxy 0.5B sous-optimal) et valide le lissage isotonic (§17, alors non soumis faute de convergence avec ce travail) sur les nouveaux signaux.

### Résultat 2 — vraisemblance exacte Gumbel-Max/TextSeal : négatif partout

Testé en ajoutant les émissions `gumbel_exact`/`textseal_exact` (LLR fermé sur `logp_target`) aux émissions binnées existantes (logsumexp = hypothèses alternatives), avec plusieurs variantes de calibration :

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| binned50 (référence, sans exact) | 0.3484 |
| + exact (p_min=1e-4, clip=8, défauts) | 0.2790 |
| exact **seul** (binned retiré) | 0.2478 |
| + exact, p_min=1e-3 / 1e-5 | 0.2793 / 0.2802 |
| + exact, clip=4 / clip=12 | 0.3026 / 0.2781 |
| entbin5+iso (référence, sans exact) | 0.4301 |
| entbin5 + exact | 0.3510 |
| entbin5+iso + exact | 0.3573 |

**Verdict : négatif dans toutes les variantes testées (6 combinaisons p_min/clip + isolé + combiné) — rejeté.** Hypothèse la plus probable : le modèle théorique suppose un tirage `argmax` pur sur les logits bruts, alors que la génération réelle utilise vraisemblablement **temperature/top-p sampling** (mentionné comme limite connue dans le commentaire de `Emission.p_min`) — `p_target` (probabilité softmax brute) n'est alors plus la vraie probabilité de sélection sous H0, cassant l'hypothèse `r | H0 ~ Uniform(0,1)` implicite au calcul de LLR. Le signal empirique binné (§11), qui n'a besoin d'aucune hypothèse sur le mécanisme d'échantillonnage, reste strictement supérieur.

**Ne pas réinvestir sur cette piste** sans informations supplémentaires sur les paramètres de génération (température, top-p) — actuellement inconnus.

### Soumission

`cv_smm.py --final b50_ps2_elo_entbin5_iso` (entropie 7B + isotonic, **sans** vraisemblance exacte) → **#994, public 0.349** (+6,3 % rel. vs #771). Gain public plus modeste que le gain CV (+16,7 %) — cohérent avec le pattern déjà documenté (§10, §16) où les gains CV ne transfèrent jamais à 100 % au public, sans que ce soit un signal de sur-ajustement (ratio public/CV reste dans la fourchette observée historiquement).

**Format vérifié avant soumission :** 1320/1320 docs, scores ∈ [0.446, 0.569] (plage resserrée autour de 0.5, cohérente avec les soumissions précédentes).

---

## 19. Recherche externe + fermeture de la piste "vraisemblance exacte" pour KGW/Unigram (négatif, bug trouvé et corrigé)

Suite à la demande d'analyser en profondeur l'écart avec la tête du leaderboard (public 0.32 alors vs meilleure équipe 0.46) : audit complet de ce document, recherche de publications sur la localisation de watermark en texte mixte, et fermeture de la piste `kgw_lpg`/`unigram_lpg` laissée ouverte en §18 ("pas encore câblée dans le scorer").

### 19.1 Recherche bibliographique — aucun algorithme publié supérieur trouvé pour ce cadre

Recherche ciblée sur la localisation de watermark en texte mixte (au-delà des 4 papiers déjà assignés) :

| Source | Apport |
| ------ | ------ |
| Code officiel TextSeal (`vendor/textseal/textseal/watermarking/detector.py::localized_detect`) | Référence organisateurs : *geometric cover search* (scan statistique dyadique) + *boundary smoother* (moyenne mobile seuillée) → labels binaires, pas de score continu par token. Confirme que notre dédup `(contexte+token)` est **identique** à leur `dedup_key` (pas de bug de dédup). |
| Zhao et al., "Efficiently Identifying Watermarked Segments in Mixed-Source Texts", ACL 2025 (GCD + AOL/Aligator) | Algorithme de référence académique le plus proche de notre tâche (KGW/Unigram/Gumbel mixés). AOL = lissage en ligne adaptatif multi-échelle (pas de longueurs canoniques supposées). **Notre SMM est structurellement plus fort dans notre cadre** car on connaît les longueurs de spans quasi-discrètes (information que ni GCD/AOL ni WISER n'exploitent) — marginaliser exactement sur ces longueurs via forward-backward domine un lissage non-paramétrique générique. |
| Karmakar et al., "WISER" (epidemic change-points) | Même constat : détection de segments sans prior de longueur. Pertinent si on perdait la structure de longueurs canoniques (pas notre cas). |
| `WatermarkBase._get_greenlist_ids` (vendor `lm-watermarking`) | Vérifié `select_green_tokens=True` (valeur YAML) ⇒ `greenlist = perm[:greenlist_size]` — **exactement** ce que `kgw_scores.py` et `entropy_pass.py` implémentent déjà. Pas de bug de ce côté. |

**Conclusion : l'écart au leaderboard n'est probablement pas dû à un algorithme de localisation publié qu'on aurait manqué** — notre approche SMM (posterior exact, prior de longueur fitté) est déjà plus adaptée à ce cadre précis que les méthodes de la littérature généraliste. L'écart vient plus vraisemblablement de la précision de calibration des signaux bruts (cf. 19.2) et/ou de données d'entraînement supplémentaires (piste synthétique, en cours par l'agent parallèle).

### 19.2 KGW/Unigram "vraisemblance exacte" via `kgw_lpg`/`unigram_lpg` : bug trouvé + corrigé, résultat final négatif

Le pass GPU 7B (§18) calcule aussi `kgw_lpg`/`unigram_lpg` = log P(vert | contexte) sous la distribution **boostée** (delta/strength appliqué avant softmax), mais ces signaux n'étaient pas encore câblés dans `smm_scorer.py`. Motivation : le Bernoulli global KGW plafonne à 9 % de spans détectés (§17) — un LLR fermé par position, comme pour Gumbel/TextSeal, pourrait faire beaucoup mieux sans aucun fit.

**Câblage initial (naïf) : effondrement catastrophique.** Première implémentation : `LLR = log(p_vert_boosté / gamma)` si vert réalisé, `log((1-p_vert_boosté)/(1-gamma))` si rouge — référence H0 = constante `gamma` (comme pour le Bernoulli fitté). Résultat : **CV s'effondre à 0.002–0.024** (vs 0.4301 baseline). Diagnostic (`corrcoef` sur les pools H0/H1 labellisés) : `corr(bit vert réalisé, p_vert_boosté) = 0.69 sur les tokens CLEAN (label=0)`, alors qu'elle devrait être nulle si `gamma` était la bonne référence. **Cause : aux positions où le modèle est déjà confiant (faible entropie) et où son token dominant naturel est vert, `p_vert_boosté` est proche de 1 — boosté ou non — donc même un token clean (jamais boosté) y est très souvent vert. Utiliser `gamma` comme H0 crédite à tort ces positions confiantes comme preuve de watermark, indépendamment de la présence réelle d'un watermark.**

**Correction :** inverser la transformation boost pour retrouver la probabilité **non-boostée** `g0` à cette position (`g0 = p_boosté / (p_boosté + (1-p_boosté)·e^boost)`, dérivé de la définition KGW `p_boosté = g0·e^δ / (g0·e^δ + (1-g0))`), puis comparer boosté vs non-boosté à la **même** position :
`LLR = log(p_boosté/g0)` si vert, `log((1-p_boosté)/(1-g0))` si rouge. Vérifié empiriquement : `mean(g0) = 0.2507` sur les tokens H0 (récupère bien gamma en moyenne, confirmant la formule), et le LLR corrigé est borné (p99 ≈ 1.26, pas d'explosion).

**Résultat final (LLR corrigé, CV 5-fold) :**

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| `b50_ps2_elo_entbin5_iso` (référence, Bernoulli fitté) | **0.4301** |
| + KGW exact (LLR corrigé, remplace le Bernoulli KGW) | 0.4031 |
| + Unigram exact (LLR corrigé, remplace le Bernoulli Unigram) | 0.4243 |
| + KGW exact + Unigram exact | 0.3994 |

**Verdict : négatif dans les trois variantes, même après correction du bug.** Le Bernoulli fitté par bin d'entropie (§16) capture déjà, de façon empirique et robuste, l'essentiel du signal exploitable ; le détecteur fermé par position — plus bruité (inversion numérique instable près de p→0/1, `KGW_VOCAB_SIZE` et `delta` pinnés mais non re-vérifiés à cette précision) — n'apporte pas de gain net. **Non retenu.**

### 19.3 Vraisemblance exacte Gumbel-Max/TextSeal : confirmation que la dilution n'est pas qu'un effet de poids uniforme

Complément à la piste négative de §18.2 : test d'un poids de mixture réduit (`SmmParams.mix_log_weights`, mécanisme de §17.C) sur les hypothèses `gumbel_exact`/`textseal_exact` plutôt que le poids uniforme `1/n_hyp` déjà écarté.

| Poids exact (vs 1.0 pour les autres hypothèses) | CV TPR@0.1%FPR |
| ------------------------------------------------ | -------------- |
| 1.0 (uniforme, déjà testé §18.2) | 0.3573 |
| 0.3 | 0.3731 |
| 0.1 | 0.3849 |
| 0.03 | 0.3986 |
| 0 (= référence sans exact) | **0.4301** |

**Confirme que ce n'est pas un problème de poids de mixture** : même à poids quasi nul (0.03), le signal exact dégrade encore le CV de 3 points. Le diagnostic de §18.2 (mismatch température/top-p à la génération, cassant l'hypothèse `r|H0 ~ Uniform(0,1)`) est la bonne explication — aucun poids de mixture ne peut compenser un LLR structurellement biaisé. **Confirme la recommandation de §18.2 : ne pas réinvestir sur cette piste sans calibration température/top-p au préalable** (ex. fit par MLE sur les spans H1 labellisés, déjà évoqué comme piste par l'agent parallèle).

### 19.4 Conclusion et pistes restantes par ROI

Toutes les pistes testées dans cette section (exact KGW, exact Unigram, poids réduit sur exact Gumbel/TextSeal) sont **négatives et documentées pour éviter qu'elles soient retestées sans nouvelle information**. Le meilleur pipeline reste `b50_ps2_elo_entbin5_iso` (CV 0.4301, public 0.349, soumission #994).

Pistes restantes non explorées ici, par ROI décroissant :

1. ~~**Calibration température/top-p du détecteur exact**~~ → **testé §20.1, négatif** — ne pas réinvestir.
2. **Données synthétiques massives** — code prêt (`gen_synthetic.py`, §20.2) ; job GPU JURECA en attente.
3. ~~**Couverture géométrique multi-échelle**~~ → **testé §20.3, négatif sur public** — ne pas réinvestir.
4. **Vérification indépendante de `KGW_VOCAB_SIZE=151665` et `delta=1.5`** au-delà du probe déjà fait (§8).

**Fichiers modifiés :** `smm_scorer.py` (`green_exact_llr`, kinds `kgw_exact`/`unigram_exact`, `Emission.boost`), `fit_smm.py` (`use_exact_kgw`, `use_exact_unigram`, `exact_mix_weight`), `cv_smm.py` (`load_lpg`, configs `*_kgwx`/`*_unix`/`*_bothx`/`*_allx`/`*_exw*`).

---

## 20. Points 1–3 (session 2026-07-03) : calibration exact, synthèse, couverture géométrique

Traitement autonome des trois pistes §19.4. Résumé : **Point 1 fermé (négatif)** ; **Point 3 testé et rejeté sur public** ; **Point 2 code prêt, génération GPU bloquée par accès JURECA**.

### 20.1 Point 1 — Calibration température / top-p pour l'exact LLR (négatif, piste fermée)

**Hypothèse :** la génération réelle applique softmax(logits/T) + nucleus top-p avant le Gumbel-max ; utiliser `p_target = softmax(logits)[token]` (T=1, pas de top-p) fausse le LLR Aaronson.

**Implémentation :** `calib_pass.py` — une passe 7B par document, grille `(T, top_p) ∈ {0.8,0.9,1.0} × {0.9,0.95,1.0}` → `output/calib_T{t}_p{p}_{split}.npz`. Job SLURM **15402077** (train+val, ~30 s). `calib_search.py` : stage 1 = exact seul (`binned50_edge_lo_ps2_exactonly`), stage 2 = exact dans le pipeline complet.

**Résultats CV (exact isolé, TPR@0.1%FPR) :**

| Tag (T, top_p) | CV |
| -------------- | --- |
| T0.8_p0.9 | 0.0727 |
| T0.8_p0.95 | 0.1095 |
| T0.8_p1.0 | 0.1929 |
| T0.9_p0.9 | 0.0997 |
| T0.9_p0.95 | 0.1552 |
| T0.9_p1.0 | 0.2267 |
| T1.0_p0.9 | 0.1368 |
| T1.0_p0.95 | 0.2057 |
| **T1.0_p1.0** | **0.2478** |
| raw `logp` (identique) | 0.2478 |

**Stage 2 (pipeline complet) :**

| Config | CV |
| ------ | --- |
| `b50_ps2_elo_entbin5_iso` (sans exact) | **0.4301** |
| + exact calibré T1.0_p1.0 | 0.3573 |

**Verdict :** la calibration T/top-p **ne débloque pas** l'exact LLR. Le meilleur cas calibré = logits bruts (T=1, pas de top-p), confirmant que le mismatch n'est pas simplement « oublier top-p » — probablement d'autres paramètres de génération cachés ou un mécanisme différent. **Ne pas réinvestir.**

### 20.2 Point 2 — Données synthétiques (code prêt, job GPU en attente)

**Implémentation :** `gen_synthetic.py` — génération réelle via les 4 générateurs vendor (GumbelmaxGenerator, TextSealGenerator, WatermarkLogitsProcessor KGW, GPTWatermark Unigram), contexte clean 32 tokens tiré du train/val, 6 longueurs canoniques × 4 schémas × N docs/cellule. Self-check z par schéma en fin de run. Entropie 7B enregistrée pendant la génération → `entropy_synth.npz`.

**Intégration fit :** `fit_smm.py` (`synthetic_docs=`), `cv_smm.py` (`load_synthetic`, config `b50_ps2_elo_entbin5_iso_synth`, flag `use_synthetic=True`).

**Statut compute :** scripts SLURM `run_synth_smoke.sh` (devel, n=3/cellule) et `run_synth.sh` (prod, n=30/cellule → 720 docs synth). **Non exécutés** — accès JURECA interrompu (master SSH stale, TOTP consommé sans auth). À relancer dès reconnexion.

**Commande prod (JURECA) :**
```bash
cd task_1_text_watermark/alexandre && sbatch run_synth.sh
# puis CV : python cv_smm.py --configs b50_ps2_elo_entbin5_iso b50_ps2_elo_entbin5_iso_synth
```

### 20.3 Point 3 — Couverture géométrique multi-échelle (négatif sur public)

**Implémentation :** `multiscale.py` — fenêtres dyadiques (stride L/2, min_len=24, pattern TextSeal `_geometric_cover_search`), boost du log-odds SMM : `log_odds += w * max_window_z`. Option boundary smoother (moving average). Paramètres dans `SmmParams` : `multiscale_weight`, `boundary_window`, etc.

**Résultats CV 5-fold :**

| Config | CV TPR@0.1%FPR |
| ------ | -------------- |
| `b50_ps2_elo_entbin5_iso` (baseline) | **0.4301** |
| + multiscale w=0.2 (`_ms02`) | 0.4315 |
| + multiscale w=0.5 | 0.4163 |
| + boundary w=20 | 0.4305 |
| + ms0.2 + boundary | 0.4345 |

Légère amélioration CV avec w=0.2 (+0.0014 absolu). **Soumission #1513** (`_ms02`) → **public 0.3364** (−0.0126 vs #994). Sur-ajustement val / dégradation test — **rejeté**. Le SMM forward-backward avec prior de longueurs canoniques capture déjà l'essentiel de la localisation ; le boost géométrique post-hoc ajoute du bruit au seuil 0.1% FPR.

### 20.4 Fichiers ajoutés / modifiés (session 2026-07-03)

| Fichier | Rôle |
| ------- | ---- |
| `calib_pass.py`, `run_calib.sh`, `calib_search.py` | Point 1 — grille T/top-p |
| `gen_synthetic.py`, `run_synth.sh`, `run_synth_smoke.sh` | Point 2 — génération synthétique |
| `multiscale.py` | Point 3 — couverture géométrique |
| `smm_scorer.py` | boost multiscale/boundary sur log-odds |
| `fit_smm.py` | `synthetic_docs`, params multiscale |
| `cv_smm.py` | configs `_ms*`, `_bnd*`, `_synth`, `load_synthetic`, `load_p_target_tag` |

**Prochaine action prioritaire :** reconnecter JURECA → `sbatch run_synth.sh` → CV avec `b50_ps2_elo_entbin5_iso_synth`. Pipeline prod reste #994 (`b50_ps2_elo_entbin5_iso`).
