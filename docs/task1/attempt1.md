# Tâche 1 — Attempt 1 (Alexandre)

**Responsable :** Alexandre (ansart1)  
**Code :** [`task_1_text_watermark/alexandre/`](../../task_1_text_watermark/alexandre/)  
**Spec :** [`docs/subject/task_1.md`](../subject/task_1.md) · **Métrique :** [`docs/task1/scoring.md`](scoring.md)

---

## État actuel (2026-07-02)

| Élément | Statut |
|---|---|
| Pipeline HMM (approche retenue) | ✅ implémenté et testé |
| Soumission API **#30** (3 détecteurs, sans KGW) | ✅ acceptée — score public **0.0065** |
| Précalcul KGW sur JURECA (job `15399747`) | ✅ `kgw_{train,validation,test}.npz` sur cluster |
| Soumission avec KGW (job `15399857`) | ⏳ job SLURM lancé — score API **en attente** |
| Val locale TPR @ 0.1% FPR (sans KGW, scores flottants) | **0.0185** |

**Meilleur score leaderboard connu :** 0.0065 (soumission #30). La soumission avec les 4 détecteurs devrait faire mieux une fois le job `15399857` terminé.

---

## 1. Objectif et métrique

Pour **chaque token** de chaque document test, prédire un score ∈ [0, 1] : confiance que le token a été généré **pendant qu’un watermarking était actif** (label binaire 0/1 en train/val, caché en test).

La métrique leaderboard n’est **pas** une régression MSE : c’est **TPR @ 0.1% FPR** sur **tous les tokens poolés** (voir [`scoring.md`](scoring.md)). Il faut classer les tokens watermarkés au-dessus de quasi tous les tokens clean, avec un budget de faux positifs très faible (1 clean sur 1000 max).

**Quatre familles** possibles dans le dataset (souvent mélangées dans un même document) : TextSeal, Gumbel-Max, Unigram, KGW. On ne prédit pas *quelle* famille — seulement *watermark actif ou non*.

---

## 2. Infrastructure

| Ressource | Chemin |
|---|---|
| Dataset (local) | `cispa_final/data/watermark_localization/{train,validation,test}.jsonl` |
| Dataset (cluster) | `/p/scratch/training2625/ansart1/loki/watermark_localization/` |
| Clés + params détecteurs | [`task_1_text_watermark/watermark_config.yaml`](../../task_1_text_watermark/watermark_config.yaml) |
| Repos vendor (commits épinglés) | `task_1_text_watermark/vendor/{textseal,lm-watermarking,unigram-watermark}/` |
| API | `CISPA_BASE_URL` + `CISPA_API_KEY` dans `.env` (cluster) · task id **`30-watermark-localization`** |
| Soumission / historique | `shared/submit.py` → `history/submissions.jsonl` |
| Eval locale + viewer | `shared/task1_eval.py` → `history/task1_viz/*.json` |

**Tokenizer :** `Qwen/Qwen2.5-7B-Instruct`. Utiliser les `token_ids` du JSONL tels quels — ne pas retokenizer le champ `text`.

**Volumes :** train 90 docs · val 90 docs · test 1 320 docs (~1,4 M tokens en test).

---

## 3. Pipeline actuel (vue d’ensemble)

```
token_ids (JSONL)
       │
       ├─► detectors.py ──► signaux CPU : TextSeal, Gumbel-Max, Unigram
       │
       └─► kgw_scores.py (GPU JURECA, une fois) ──► masques green/red par token
                │
                ▼
         hmm_scorer.py
           · fit émissions LLR sur train (spans → schéma par heuristique z)
           · forward-backward → P(watermark actif) par token
                │
                ▼
         run_hmm.py → validation_scores.jsonl + submission.jsonl
                │
                ├─► task1_eval.py (val, si labels)
                └─► submit.py (test → leaderboard)
```

**Approche retenue :** modèle de Markov caché (HMM) — voir section 5. Les approches fusion z-score (`build_scores.py`) et régression logistique (`train_calibrator.py`) ont été testées puis **abandonnées** (sections 6).

---

## 4. Signaux par token (`detectors.py`)

Chaque détecteur produit **une statistique par position** `t`, à partir des `token_ids` uniquement (pas de forward pass du LLM).

| Schéma | Signal | Clés / params (YAML) | H0 (attendu) |
|---|---|---|---|
| **TextSeal** | Dual-key Gumbel : `α·g(r_a)+(1-α)·g(r_b)`, `g(r)=-log(1-r)` | `key_a`, `key_b`, `ngram=3`, `α=0.5` | μ=1, σ²=0.5 |
| **Gumbel-Max** | `-log(1-r)` sur PRF uniforme | `secret_key`, `ngram=2` | μ=1, σ²=1 (Exp(1)) |
| **Unigram** | 1 si token ∈ greenlist fixe | `watermark_key`, `fraction=0.5`, `vocab_size=151643` | Bernoulli(0.5) |
| **KGW** | *non calculé ici* — voir `kgw_scores.py` | `gamma=0.25`, schéma `ff-anchored_minhash_prf-4-True-1306382177` | Bernoulli(0.25) |

**Implémentation PRF :** chargement direct de `vendor/textseal/textseal/watermarking/core.py` (`prf_uniform`) via `importlib`, sans importer le package `textseal` complet (évite `nltk`, etc.).

**Validation train (moyennes span wm vs clean) :**

```
textseal  : wm ≈ 1.12   clean ≈ 1.01
gumbelmax : wm ≈ 1.27   clean ≈ 1.01
```

TextSeal et Gumbel-Max sont **correctement répliqués** (signal réel, séparation modeste token à token).

**Unigram — point ouvert :** le `vocab_size` exact n’est pas dans le YAML. Avec `151643` (Qwen2.5 standard), la séparation est quasi nulle sur les spans « résiduelles ». Scan de tailles voisines : signaux faibles, pas de confirmation forte. Le HMM n’assigne que **~199 tokens** au pool d’entraînement Unigram → contribution limitée aujourd’hui.

---

## 5. Précalcul KGW (`kgw_scores.py` + job SLURM)

KGW **doit** utiliser `torch.randperm` sur un `torch.Generator` **CUDA** (Philox). Sur CPU, les greenlists ne correspondent pas à la génération organisateur → ~75 % des tokens KGW semblent « clean ».

**Job terminé :** `15399747` (`run_kgw.sh`, 1× A100, ~6 min train+val + ~5 min test).

| Fichier produit | Contenu |
|---|---|
| `output/kgw_train.npz` | `document_id → float32[m]` (0 ou 1 par token scorable ; 0.25 avant fenêtre) |
| `output/kgw_validation.npz` | idem |
| `output/kgw_test.npz` | idem (~602 KB) |

**Algorithme :** réplique du détecteur `lm-watermarking` — n-grams de longueur 4 (self-salt, Algorithm 3), greenlist via `torch.randperm(vocab_size)` sur GPU. `vocab_size` par défaut **151936** (config Qwen2.5) ; mode `--probe` compare 5 candidats sur train labellé.

**Chemin cluster :** `/p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre/output/kgw_*.npz`

---

## 6. Modèle HMM (`hmm_scorer.py`) — cœur de la soumission

### Pourquoi un HMM ?

Les stats brutes par token sont bruitées ; la métrique à FPR = 0.1 % pénalise les **pics isolés** et favorise des **spans contigus** bien localisés. Un HMM impose des transitions rares clean ↔ watermark sans lissage manuel.

### Structure

- **États :** `clean` + un état par schéma dont le signal est disponible (`textseal`, `gumbelmax`, `unigram`, `kgw` si `.npz` chargé).
- **Émissions :**
  - TextSeal / Gumbel-Max (continus) : **LLR non paramétrique** — 40 bins quantiles, densités H1 vs H0 fitées sur train.
  - Unigram / KGW (binaires) : LLR Bernoulli fermé.
  - État `clean` : émission 0 (log-likelihood nulle).
- **Transitions (fixes, non apprises) :** `p_enter=0.005` (clean → schéma wm), `p_exit=0.010` (schéma → clean).
- **Entraînement des émissions :** chaque span watermarkée du train est assignée au schéma avec le **z-score span le plus élevé** (seuil `z ≥ 3`). Les tokens de cette span alimentent le pool H1 du schéma gagnant ; tous les tokens clean alimentent H0.

**Pools H1 fités (train, sans KGW) :** TextSeal 11 915 · Gumbel-Max 15 390 · Unigram 199 · KGW 0.  
Avec les `.npz` KGW, un 4ᵉ état est ajouté automatiquement au fit.

### Inférence

Forward-backward en log-espace → **score token = 1 − P(état clean | séquence)**.

**Résultat val (sans KGW, scores flottants) :** TPR @ 0.1% FPR = **0.0185** (vs ~0.0016 fusion z naïve, vs ~0.007 régression logistique).

### Bug connu — arrondi à 6 décimales

`run_hmm.py` écrit `round(score, 6)` dans le JSONL. À FPR = 0.1 %, cela **dégrade** la métrique : val passe de **0.0185 → 0.0140** sur le même modèle. La soumission #30 a été envoyée avec cet arrondi. **À corriger** avant la prochaine resoumission (scores float64 pleine précision).

---

## 7. Soumissions et scores

| # | Méthode | KGW | Val TPR@0.1%FPR | Score API (public) | Fichier |
|---|---|---|---|---|---|
| 30 | HMM forward-backward, 3 PRF | ❌ | 0.0185 (float) / 0.0140 (arrondi) | **0.0065** | `output/submission.jsonl` |
| (en cours) | HMM + 4 détecteurs | ✅ | *job `15399857`* | *en attente* | idem (écrasé à la fin du job) |

Soumission #30 : validée par `task_template.validate_predictions`, envoyée via `shared/submit.py`. Traces : `history/submissions.jsonl`, viewer `history/task1_viz/hmm_no_kgw_v1.json`.

**Lecture des scores :** un ranker aléatoire donne TPR@0.1%FPR ≈ 0.001 par construction. **0.0065 ≈ 6,5× le hasard** — signal réel mais modeste ; l’absence de KGW (~42 % des spans train non expliquées par TextSeal/Gumbel) plafonne fortement le rappel.

---

## 8. Approches abandonnées (référence)

| Approche | Fichiers | Val TPR@0.1%FPR | Pourquoi abandonnée |
|---|---|---|---|
| Fusion z-score + sigmoïde | `build_scores.py` | ~0.0016 | Pas de structure de span, seuil arbitraire |
| Régression logistique multi-features | `features.py`, `train_calibrator.py` | ~0.007 | Classifieur token-à-token, pas de transitions |

Conservées dans le repo pour comparaison ; **ne pas soumettre**.

---

## 9. Limites actuelles et prochaines étapes

**Par impact attendu :**

1. **Soumission avec KGW** — job `15399857` ; levier principal (~¼ des tokens wm ≈ KGW pur aujourd’hui ignorés).
2. **Supprimer l’arrondi `round(..., 6)`** dans `run_hmm.py` — quick win mesuré (−25 % TPR relatif sur val).
3. **Confirmer `vocab_size` Unigram** — tokenizer HF sur cluster ou alignement avec la greenlist organisateur.
4. **Baum-Welch** — apprendre `p_enter` / `p_exit` au lieu de valeurs manuelles.
5. **Pondération entropie TextSeal** — forward pass `Qwen2.5-7B-Instruct` (GPU) comme dans le paper.
6. **Assignation span→schéma souple** (EM) au lieu du cutoff `z ≥ 3`.
7. **Cross-val** train+val (180 docs) pour estimer TPR@0.1%FPR moins bruité.
8. **Couverture géométrique** (`localized_detect` vendor TextSeal) pour spans diluées.

---

## 10. Comment exécuter

### Local (CPU) — sans KGW

Prérequis : dataset dans `data/watermark_localization/`, Python avec `torch`, `numpy`, `scipy`, `scikit-learn`.

```bash
cd cispa_final/task_1_text_watermark/alexandre

python run_hmm.py \
  --data-dir ../../data/watermark_localization \
  --out-dir output \
  --splits validation test

python ../../shared/task1_eval.py \
  --dataset ../../data/watermark_localization/validation.jsonl \
  --predictions output/validation_scores.jsonl \
  --method "HMM: TextSeal+GumbelMax+Unigram (no KGW)" \
  --out hmm_no_kgw_v1

python ../../shared/submit.py output/submission.jsonl \
  --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "HMM forward-backward (no KGW)"
```

### JURECA — pipeline complet avec KGW

**Compte :** `training2625` · **Réservation :** `cispahack` · **Partition :** `dc-gpu`

```bash
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre

# Étape 1 — précalcul KGW (GPU, ~10 min) — DÉJÀ FAIT (job 15399747)
sbatch run_kgw.sh
# → output/kgw_{train,validation,test}.npz

# Étape 2 — HMM + eval val + submit API (job 15399857 ou manuel)
sbatch run_hmm_submit.sh
```

**Exécution manuelle** (sur nœud compute, pas login node — conflit venv/module PyTorch) :

```bash
module load GCC CUDA PyTorch torchvision
source /p/scratch/training2625/ansart1/loki/watermark_localization/.venv/bin/activate
export PYTHONPATH=/p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre:...

python run_hmm.py \
  --data-dir /p/scratch/training2625/ansart1/loki/watermark_localization \
  --kgw-dir output \
  --out-dir output \
  --splits validation test
```

---

## 11. Carte des fichiers

| Fichier | Rôle |
|---|---|
| **`detectors.py`** | Signaux PRF CPU : TextSeal, Gumbel-Max, Unigram |
| **`kgw_scores.py`** | Masques green KGW (CUDA Philox obligatoire) |
| **`hmm_scorer.py`** | Fit LLR + forward-backward HMM |
| **`run_hmm.py`** | CLI principale : fit train → score val/test → JSONL |
| **`run_kgw.sh`** | SLURM : précalcul KGW |
| **`run_hmm_submit.sh`** | SLURM : HMM + eval + `submit.py` |
| `build_scores.py` | *(abandonné)* fusion z-score |
| `features.py`, `train_calibrator.py` | *(abandonné)* régression logistique |

**Sorties :** `alexandre/output/submission.jsonl`, `validation_scores.jsonl`, `kgw_*.npz`, logs SLURM dans `alexandre/logs/`.
