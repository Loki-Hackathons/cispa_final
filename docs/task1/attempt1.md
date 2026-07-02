# Tâche 1 — Attempt 1 (Alexandre)

**Responsable :** Alexandre (ansart1)
**Code :** [`task_1_text_watermark/alexandre/`](../../task_1_text_watermark/alexandre/)
**Spec :** [`docs/subject/task_1.md`](../subject/task_1.md)

---

## État actuel (2026-07-02, soir)

| Élément | Statut |
| -------- | ------ |
| Pipeline HMM (v1, abandonné) | ❌ remplacé — boundary bleed, score plafonné |
| Pipeline semi-Markov (v2, **retenu**) | ✅ implémenté, testé, soumis |
| Soumission #30 (HMM, sans KGW) | score public **0.0065** |
| Soumission #101 (HMM + KGW) | score public **0.0071** |
| Soumission #159 (semi-Markov, sans KGW) | score public **0.2001** — ×28 vs #101 |
| Soumission #262 (semi-Markov + KGW) | score public **0.2029** |
| Soumission #408 (semi-Markov + KGW + Unigram) | score public **0.2037** |
| **Soumission #509 (CV-binned : SMM + LLR empiriques + priors fittés)** | **score public 0.2526** — **best actuel** (+24 % rel. vs #408) |
| Val TPR@0.1%FPR — semi-Markov v2 (+ KGW + Unigram) | **0.3137** (neutre vs +KGW seul) |
| KGW | ✅ intégré via `.npz` précalculés (CUDA Philox, JURECA) |
| Unigram | ✅ intégré (`perm@152064`, éligibilité `id < 151643`) — gain marginal, plafond atteint (§9) |
| CV 5-fold + LLR binned (`cv_smm.py`, `fit_smm.py`) | ✅ fait et soumis — CV 0.3484 vs 0.2764 baseline, public **0.2526** (§11) |

**Meilleur score leaderboard public : 0.2526** (CV-binned). Le gros du gain vient (1) du passage HMM → semi-Markov à longueurs discrètes + fix du bug n-grammes répétés, (2) du remplacement des émissions gaussiennes à shifts fixes par des LLR empiriques binnés + priors fittés, sélectionnés en CV 5-fold. KGW (+1.4 % relatif) et Unigram (+0.4 % relatif) ajoutent chacun un gain modeste mais net.

**Pipeline de production actuel :** `run_smm.py` + `--kgw output/kgw_{split}.npz` — Unigram est **toujours actif** dans `smm_scorer.py` (4ᵉ signal). Les shifts gaussiens par schéma sont les defaults du module (reproduisent #408).

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

**Note Unigram :** la config retenue n'inclut **pas** l'émission Unigram. Testé explicitement (`binned50_edge_lo_ps2_uni`) : rebrancher Unigram dans le pipeline CV-binned fait passer le CV de 0.3484 à 0.3408. Cohérent avec le plafond de signal (§9/§14) : ~1 doc test réellement attribuable, contre un coût FPR diffus sur tous les autres. Le pipeline #408 (gaussien + Unigram) reste reproductible via `run_smm.py`, mais le best actuel (#509) est sans Unigram.

### Soumission

Refit sur les 180 docs, scoring test (`cv_smm.py --final binned50_edge_lo_ps2`), soumis → **#509, public 0.2526** (+24 % relatif vs 0.2037). Le gain CV (+26 % relatif) transfère presque entièrement au public — contrairement aux réglages sur val seule (§10), ce qui valide le CV comme métrique de décision.

**Vérifications faites avant soumission :** refactor de `smm_scorer.py` validé par test de régression (scores identiques à #262 à < 1e-12) ; config gagnante revalidée avec un second découpage de folds (seed 1 → 0.3516, stable) ; format de soumission validé (1320 docs, scores ∈ [0,1]).

---

## 12. Prochaines étapes, par impact attendu

1. **Pondération par entropie** (forward pass Qwen sur GPU, papier TextSeal) — concentre le signal sur tokens à haute incertitude.
2. **Couverture géométrique** (`localized_detect` vendor TextSeal) — spans diluées dans docs mixtes.
3. **Raffiner la grille CV** autour de la config retenue (bins, clip, edge) — rendement décroissant attendu, le levier principal est consommé.
4. ~~Shifts faibles / priors à la main~~ — **testé, négatif** (§11).
5. ~~Unigram~~ — **résolu, plafond atteint** (§9).
6. ~~Hypothèse vocab KGW~~ — **écartée** (§8).

---

## 13. Comment exécuter

### Pipeline best actuel (public 0.2526) — CV-binned

```bash
cd cispa_final/task_1_text_watermark/alexandre

python cv_smm.py --grid                                   # (re)valider la grille en CV
python cv_smm.py --final binned50_edge_lo_ps2 --out submission_cv.jsonl
python ../../shared/submit.py submission_cv.jsonl \
  --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "SMM + binned LLR + fitted priors (CV-selected)"
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

## 14. Carte des fichiers

| Fichier | Rôle |
| ------- | ---- |
| `detectors.py` | Signaux PRF CPU (TextSeal, Gumbel-Max) + dédup n-grammes |
| `unigram_scan.py` | Greenlist Unigram (perm 152064) + scan non supervisé |
| `smm_scorer.py` | **Modèle retenu** : semi-Markov, 4 signaux (TS+GM+KGW+Unigram) |
| `run_smm.py` | CLI scoring → JSONL, `--kgw` pour KGW |
| `fit_smm.py` | Fit supervisé priors + LLR binned sur train |
| `cv_smm.py` | CV 5-fold document-level + `--final` pour soumission |
| `kgw_scores.py` | Masques KGW (CUDA Philox, JURECA) |
| `run_kgw.sh` | SLURM précalcul KGW |
| `../../scripts/fetch_kgw.sh` | Rapatriement `.npz` depuis JURECA |
| `sweep_shifts.py`, `sweep_prior.py` | Sweeps shifts/priors (train+val) |
| `hmm_scorer.py`, `run_hmm.py` | *(abandonné v1)* |
| `matched_filter.py`, `build_scores.py`, `train_calibrator.py` | *(abandonnés)* |

**Sorties soumises :** `submission_smm_cv.jsonl` (**0.2526, best**), `submission_smm_kgw_uni.jsonl` (#408, 0.2037), `submission_smm_kgw.jsonl` (#262, 0.2029), `submission_smm_nokgw.jsonl` (#159, 0.2001). `submission_smm_cvbinned.jsonl` = même config `binned50_edge_lo_ps2` regénérée indépendamment (non soumise, redondante).
