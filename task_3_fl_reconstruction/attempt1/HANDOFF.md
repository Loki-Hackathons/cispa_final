# Task 3 — Passation (FL Data Reconstruction)

Dernière mise à jour : 2026-07-03, ~00:15. Auteur sortant : Bastian (paoli1).
À lire en entier avant de toucher au code. Complète (ne remplace pas) `TASK3_GUIDE.md`.

---

## 1. En une phrase

On reconstruit 1536 images (12 modèles × 128) à partir des gradients FL fuités.
Métrique = SSIM moyen après appariement 1-à-1 (chaque image soumise est appariée à
une image de vérité **distincte**, donc dupliquer une bonne reconstruction ne sert à rien).
**Meilleur score actuel : 0.255** (soumission ID 1114). Objectif : > 0.5.

---

## 2. État actuel (chiffres)

| | |
|---|---|
| Meilleur score public | **0.255** (soumission ID 1114) |
| Baseline précédent | 0.2469 |
| Régression connue à éviter | 0.2033 (ID 880, voir §5 « clamp ») |
| Fichier soumis | `submission_v2.pt` (généré par `reconstruct_v2.py`) |
| Tout le code | poussé sur `main` (GitHub Loki-Hackathons/cispa_final) |

Le split public = 30 % des données ; le classement final = 70 % cachés.
**Ne jamais régler d'hyperparamètres sur le retour du leaderboard** (risque d'overfit
du split public). On valide en synthétique (`bench_selection.py`) ou par re-simulation
du gradient observé, jamais sur le score public.

---

## 3. Pipeline (ce qui tourne)

`reconstruct_v2.py` est le point d'entrée principal (100 % CPU, ~secondes/modèle) :

- **MLP** (`recover_mlp`) : lignes analytiques `r_i = gW_i/gb_i`, sélection par
  `own_margin` (meilleur sélecteur label-free, cf. §4), clustering des lignes en
  images distinctes (`separation.isolated_recovery`), puis filtre de structure.
- **CNN** (`recover_cnn`) : on récupère les feature maps d'entrée de fc1, puis on
  **inverse les filtres conv « transmit »** canal par canal (`channels.py`) pour
  récupérer le vrai RGB. Fallback niveaux de gris si pas de transmit.
- **ViT** (9, 11) : pas de chemin analytique → remplissage prior (bruit basse-fréquence).
- **Filtre de structure** (`separation.keep_structured_and_fill`) : jette les tuiles
  ressemblant à du bruit et remplit les 128 slots avec des variantes augmentées
  (flips + jitter) des vrais visages — un visage plausible matche mieux qu'du bruit.

Commandes :
```bash
python reconstruct_v2.py --diagnose      # table recoverabilité + previews PNG (CPU)
python reconstruct_v2.py --out submission_v2.pt   # construit les 12 (CPU)
python submit.py --check submission_v2.pt         # validation format + mean_quality
```

---

## 4. Ce qu'on a fait cette session (dans l'ordre)

1. **Analyse visuelle des previews** (`output/v2_diag/v2_model*.png`, copiés dans
   `diag_share/` pour transfert git). Diagnostics :
   - CNN 2/7/10/12 : visages corrects mais **teintés en rouge** (moyenne des 8 canaux
     mélangeait signal + bruit).
   - CNN 3 : fausses couleurs rouge/cyan.
   - MLP 1/5/6/8 : beaucoup de tuiles de **bruit pur** (matchent la GT à ~0).
   - CNN 12 (sigmoid) : couleurs **psychédéliques** (voir §6, bug ouvert).

2. **Corrections poussées** (commits `390288c`, `57cd42a`, `85bec25`) :
   - `channels.py` : inversion transmit par défaut (`min_delta` 0.9 → 0.55) → les CNN
     transmit partiel (2/7/10) récupèrent le vrai RGB. Non-transmit (3) → niveaux de gris.
   - `separation.py` : `structure_score` + `keep_structured_and_fill` (filtre bruit).
   - `reconstruct_v2.py` : filtre de structure câblé dans MLP + CNN ; previews `--diagnose`
     reflètent désormais la soumission finale (avant elles montraient les clusters bruts).

3. **Soumission** → **0.255** (ID 1114), `improved: True`. Gain modeste (+0.008).

4. **Benchmark synthétique** (`bench_selection.py`, vérité terrain connue) — résultat clé :

   | Régime (trap MLP) | Oracle (plafond) | `own_margin` (actuel) | Écart |
   |---|---|---|---|
   | Mélange lourd | 0.541 | 0.338 | 0.20 |
   | Isolation partielle | 0.427 | 0.401 | 0.03 |
   | Isolation totale | 0.422 | 0.422 | 0 |

   **Conclusions décisives :**
   - La **sélection de lignes est déjà quasi-optimale** en régime isolé → inutile de
     continuer à la bricoler, ça ne rapportera rien.
   - Le **plafond analytique des MLP trap est ~0.42** même en trichant → les MLP seuls
     ne nous mèneront jamais à 0.5.
   - **Le seul chemin vers 0.5 passe par les modèles non-analytiques** : MLP lisses
     (1, 4), CNN lisse (2), ViT (9, 11) — via gradient matching GPU.

5. **Plan GPU préparé** (`sbatch --array=1,4 slurm_array.sh`) mais **bloqué** : le
   scratch `/p/scratch/training2625` n'était pas monté sur `jrlogin03` (voir §7).

---

## 5. Leçons validées (NE PAS refaire ces erreurs)

- **Normalisation : utiliser `utils.to_unit()` (min/max par image), JAMAIS `clamp(0,1)`.**
  Notre SSIM local se re-normalise, donc `clamp` vs `min/max` est **invisible en local**
  mais le vrai évaluateur utilise une plage fixe [0,1]. `clamp` a coûté 0.2469 → 0.2033.
- **Ne pas optimiser CNN/ViT avec un forward deviné.** Le gradient matching contre un
  forward CNN/ViT approximatif a cassé le leaderboard (0.1427 → 0.0635). Seul le MLP a
  un forward reconstruit exactement (`rebuild.build_mlp`) → seul le MLP est sûr à optimiser.
  C'est pourquoi `run.py` a `EXACT_FORWARD = {"mlp"}`.
- **Garde anti-régression** : `run.py --optimize` garde l'analytique si l'optimisation
  ne reproduit pas mieux le gradient observé → ne peut pas régresser. `mlp_reconstruct.py
  --refine` n'a **pas** cette garde → préférer `run.py --optimize`.
- **Valider en synthétique, pas sur le leaderboard.** `bench_selection.py` a vérité terrain.

---

## 6. État par modèle (table `--diagnose` + mean_quality)

`clusters` = images distinctes trouvées ; `conf>.5` = lignes à haute confiance ;
`transmit` = force du filtre transmit CNN (`*` = détecté).

| Modèle | Famille | Act | clusters | conf>.5 | transmit | mean_q | Diagnostic |
|---|---|---|---|---|---|---|---|
| 1 | mlp | sigmoid | 1010 | 1 | – | 0.075 | Lisse → **GPU refine** |
| 2 | cnn | tanh | 953 | 1 | 0.76* | 0.238 | Lisse → GPU / transmit tanh |
| 3 | cnn | relu | 819 | 397 | 0.59 | 0.208 | Non-transmit → gris, OK |
| 4 | mlp | tanh | 986 | 984 | – | 0.076 | Lisse → **GPU refine** |
| 5 | mlp | relu | 1024 | 1 | – | 0.080 | ReLU mais 1 seul conf → à investiguer |
| 6 | cnn | relu | 1021 | 3 | 1.00* | 0.069 | Transmit parfait mais peu isolé |
| 7 | cnn | relu | 869 | 868 | 0.69* | 0.196 | **Bon** (transmit + isolé) |
| 8 | mlp | relu | 1020 | 1017 | – | 0.066 | **Bon** analytique (plafond ~0.42) |
| 9 | vit | gelu | – | – | – | 0.162 | Bruit fill → **~0, gros gisement** |
| 10 | cnn | relu | 466 | 466 | 0.65* | 0.126 | Transmit partiel OK |
| 11 | vit | gelu | – | – | – | 0.162 | Bruit fill → **~0, gros gisement** |
| 12 | cnn | sigmoid | 1024 | 1018 | 1.00* | 0.182 | **BUG couleur** (inversion logit explose) |

> Note : `mean_quality` est un proxy contraste/bruit, **pas** le SSIM. Utile en relatif seulement.

---

## 7. Blocages d'accès rencontrés (et solutions)

- **SSH `Permission denied (publickey)` après changement de réseau** : la clé SSH avait
  une **restriction d'IP** (`from="134.96.0.0/16,..."`) sur JuDoor → ne marchait que depuis
  eduroam/CISPA. Solution : sur JuDoor → Systems → JURECA → clé SSH → mettre son IP
  publique (`x.x.x.x/32`) ou `0.0.0.0/0`, ou vider le champ.
- **`/p/scratch/training2625` inaccessible sur `jrlogin03`** : scratch non monté sur ce
  login node précis (les données ne sont **pas** perdues). Solution : se déconnecter et se
  reconnecter (`ssh juelich`) pour tomber sur un autre `jrloginXX`, puis vérifier
  `ls /p/scratch/training2625/dougnon1/Loki/ && echo OK`. Forcer un nœud :
  `ssh paoli1@jrlogin05.jureca.fz-juelich.de`.

---

## 8. ROADMAP (par priorité de gain attendu)

### Étape 0 — Débloquer l'accès (immédiat)
Reconnexion sur un login node où le scratch est monté (§7). Vérifier :
```bash
ls /p/scratch/training2625/dougnon1/Loki/cispa_final && echo OK
```

### Étape 1 — GPU sur les MLP lisses 1 et 4 (gain net attendu, SÛR)
Gradient matching à forward exact, garde anti-régression. C'est le levier le plus clair.
```bash
cd /p/scratch/training2625/dougnon1/Loki/cispa_final
git pull origin main
cd task_3_fl_reconstruction/attempt1
export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source $TASK3_DATA_ROOT/.venv/bin/activate

sbatch --array=1,4 slurm_array.sh      # écrit output/parts/model1.pt, model4.pt
squeue -u $USER
```
Quand fini :
```bash
python merge.py --parts output/parts --base submission_v2.pt --out submission_v2_gpu.pt
python submit.py --check submission_v2_gpu.pt
# soumettre submission_v2_gpu.pt (voir §9)
```

### Étape 2 — Fix couleur modèle 12 (CPU, faible risque)
CNN sigmoid transmit=1.00 mais couleurs psychédéliques : `_inv_activation` fait un logit
qui explose quand les features saturent (proche de 0/1). Piste : pour sigmoid/tanh transmit,
**ne pas inverser l'activation** — utiliser directement la sortie monotone `feats[:,o]` comme
canal (SSIM est structurel, pas besoin des valeurs exactes) puis `to_unit`. À tester d'abord
sur le preview `--diagnose` du modèle 12 (doit perdre les couleurs criardes). Voir
`channels.py::transmit_features_to_rgb` et `_inv_activation`.

### Étape 3 — ViT 9 et 11 (gros gisement mais DUR)
Actuellement remplissage bruit → ~0 SSIM (256 images à zéro). Options, par ordre de risque :
- (a) **Meilleur prior** : remplir avec des visages basse-fréquence plausibles plutôt que du
  bruit — même 0.15 de SSIM sur 256 images = +0.03 au global. Faible risque.
- (b) **Gradient matching ViT** : nécessite un forward ViT EXACT (vérifier le nommage timm
  via `python inspect_models.py --keys 9 11` et `rebuild.build_vit`). **Risqué** : un forward
  deviné a déjà cassé le score. Ne tenter qu'avec forward validé + garde anti-régression.

### Étape 4 — Denoising par moyennage de clusters (CPU, à valider en synthétique)
En régime « mélange lourd » (modèle 6 : transmit parfait mais conf>.5=3), l'oracle atteint
0.54 vs 0.34 pour nous : gros écart. Piste non testée : **baisser le seuil de clustering**
(< 0.90) pour regrouper les quasi-copies d'une même image, puis **moyenner les membres** du
cluster pour débruiter (au lieu de prendre le membre le plus net). `bench_selection.py` ne
teste PAS encore le clustering+moyennage — l'ajouter au benchmark AVANT de toucher au code
(sinon risque de régression comme le clamp).

---

## 9. Comment soumettre (rappel)
```bash
cp <fichier>.pt $TASK3_DATA_ROOT/submission.pt
cd $TASK3_DATA_ROOT
set -a && source .env && set +a
export TASK3_SUBMISSION_PATH="submission.pt"
python task_template.py           # SUBMIT=True ; cooldown 5 min entre 2 soumissions réussies
```
Alternative : `python shared/submit.py <fichier>.pt --task-id 21-fl-audit --action submit --owner paoli1`

---

## 10. Fichiers importants
- `reconstruct_v2.py` — pipeline principal (CPU)
- `channels.py` — inversion transmit CNN (fix couleur ici)
- `separation.py` — sélection `own_margin`, clustering, filtre de structure
- `bench_selection.py` — benchmark synthétique (validation sans leaderboard)
- `run.py` — orchestrateur avec `--optimize` (gradient matching MLP + garde anti-régression)
- `mlp_reconstruct.py` — refine MLP (⚠ pas de garde anti-régression)
- `slurm_array.sh` — 1 tâche GPU/modèle → `output/parts/`
- `slurm_cnn_invert.sh`, `slurm_mlp_refine.sh` — jobs GPU dédiés
- `merge.py` — assemble les parts dans une soumission
- `TASK3_GUIDE.md` — guide opérationnel complet
- `setup_cluster.sh` — met les variables d'env + venv (⚠ chemin scratch en dur)

---

# Annexe — Récapitulatif complet (historique avant + pendant v2)

> **Lecture croisée obligatoire.** Cette annexe documente tout le travail **avant** le pipeline
> `reconstruct_v2.py` (CNN invert + ViT = best **0.2469**, ID 645) **et** les leçons empiriques
> accumulées sur 12+ soumissions. Le **best actuel est 0.255** (ID 1114, `submission_v2.pt`).
> Les deux pipelines coexistent dans le repo :
>
> | Pipeline | Fichiers clés | Best score | Rôle |
> |---|---|---|---|
> | **Legacy (CNN invert + ViT)** | `cnn_invert.py`, `run.py --optimize`, `rebuild.SimpleViT` | 0.2469 (`sub_vit_both.pt`) | CNN GPU inversion, ViT gradient match — **prouvé sur le leaderboard** |
> | **v2 (analytique amélioré)** | `reconstruct_v2.py`, `channels.py`, `separation.py` | **0.255** (`submission_v2.pt`) | CPU, fix couleur transmit, sélection `own_margin`, filtre bruit |
>
> **Stratégie recommandée pour la suite :** partir de `submission_v2.pt` (0.255) comme base,
> **réintégrer** les modèles où le legacy était meilleur (CNN invert GPU pour 2/3/6/7/10/12,
> ViT 9/11 depuis `sub_vit_both.pt`), plutôt que de repartir de zéro sur l'un ou l'autre.
> Certaines leçons de l'annexe (ex. « MLP refine dégrade ») contredisent la roadmap §8 — elles
> viennent de tests **sans** garde anti-régression et **sans** le pipeline v2 ; à re-tester
> prudemment, pas à appliquer aveuglément.

## Récapitulatif complet — Task 3 (depuis le début de cette conversation)

Document de synthèse : code, stratégie, résultats, erreurs à ne pas refaire, et état historique.

### 1. Contexte du projet

| Élément | Détail |
|---|---|
| Équipe | Loki (CISPA hackathon finals) |
| Tâche | Task 3 — FL Data Reconstruction |
| Objectif | Reconstruire 1536 images (12 modèles × 128) depuis gradients + modèles white-box |
| Métrique | Mean SSIM avec matching optimal 1-vers-1 par modèle |
| Repo | `cispa_final/task_3_fl_reconstruction/attempt1/` |
| Cluster | JURECA, `training2625`, données dans `FL_Data_Reconstruction/` |
| Best score (legacy, fin phase ViT) | **0.2469** (`sub_vit_both.pt`, submission ID 645) |
| Best score (actuel, pipeline v2) | **0.255** (`submission_v2.pt`, submission ID 1114) |

On partait d'environ 0.24 en début de conversation ; on est montés à 0.2469 en finissant le ViT
(9 + 11), puis à **0.255** avec le pipeline v2 analytique amélioré.

### 2. Les 12 modèles et ce qu'on sait (stratégie legacy)

| # | Family | Activation | Shape interne | Stratégie retenue (legacy) |
|---|---|---|---|---|
| 1 | mlp | sigmoid | (3,64,64) | Analytique net.0 — ne pas toucher |
| 2 | cnn | tanh | (8,16,16) | Inversion conv (`cnn_invert`) |
| 3 | cnn | relu | (8,8,8) | Inversion + 5000 steps |
| 4 | mlp | tanh | (3,64,64) | Analytique — ne pas refine |
| 5 | mlp | relu | (3,64,64) | Analytique — ne pas toucher |
| 6 | cnn | relu | (8,64,64) | Inversion (MSE local énorme mais aide au SSIM) |
| 7 | cnn | relu | (8,16,16) | Inversion |
| 8 | mlp | relu | (3,64,64) | Analytique — ne pas toucher |
| 9 | vit | gelu | (3,64,64) | Gradient matching (cls + norm) |
| 10 | cnn | relu | (8,64,64) | Inversion |
| 11 | vit | gelu | (3,64,64) | Gradient matching (fc_norm, pas de cls) |
| 12 | cnn | sigmoid | (8,64,64) | Inversion |

Règle empirique : les batches sont distincts entre modèles — on ne réutilise pas les mêmes
images d'un modèle à l'autre.

> **Note v2 :** le pipeline `reconstruct_v2.py` remplace partiellement ces choix par une
> approche analytique unifiée (transmit filters + `own_margin`). Voir §6 du corps principal
> pour l'état v2 par modèle.

### 3. Architecture du code (attempt1/)

```
attempt1/
├── config.py           # chemins, BATCH=128, EPS, seuils dedup
├── utils.py            # IO, validate, to_unit, quality_score, dedup, TV, SSIM
├── extract.py          # extraction analytique Boenisch (MLP net.0, CNN fc1)
├── rebuild.py          # MLP/CNN/ViT exacts ou best-effort, gradient matching
├── run.py              # orchestrateur 12 modèles ; optimize gated MLP/ViT
├── cnn_invert.py       # ★ inversion CNN via vraie conv
├── fc1_analytic.py     # confiance d'isolement des lignes fc1
├── mlp_reconstruct.py  # sélection MLP + refine optionnel
├── diagnose_mlp.py     # diagnostic MLP
├── diagnose_model.py   # target vs pred CNN (model6)
├── merge.py, submit.py, analyze.py, inspect_models.py
├── slurm_cnn_invert.sh, slurm_mlp_refine.sh
├── reconstruct_v2.py   # ★ pipeline v2 (CPU, post-legacy)
├── channels.py         # ★ inversion transmit CNN (v2)
├── separation.py       # ★ clustering + own_margin + structure gate (v2)
├── bench_selection.py  # benchmark synthétique sélection (v2)
└── TASK3_GUIDE.md
```

**Rôles principaux**

| Fichier | Rôle |
|---|---|
| `extract.py` | `row_i = gW_i / gb_i` sur net.0 (MLP) ou fc1 (CNN) → candidats |
| `cnn_invert.py` | Optimise des pixels x pour `activation(conv(x)) ≈ target` |
| `rebuild.py` | Reconstruit le forward ; `gradient_match()` pour ViT/MLP refine |
| `run.py` | Pipeline par modèle : analytic → sélection 128 → optimize optionnel |
| `utils.py` | `dedup_select`, `quality_score`, validation soumission |
| `reconstruct_v2.py` | Pipeline v2 unifié (CPU) — voir §3 corps principal |

### 4. Les trois familles — comment ça marche dans le code

#### A. CNN (modèles 2,3,6,7,10,12) — cœur du score ~0.24 (legacy)

**Étape 1 — Extraction analytique (fc1)**

```python
row_i = gradient_fc1.weight[i] / gradient_fc1.bias[i]
# → feature map (8, H, W)
```

**Étape 2a — Baseline (0.1427)** : `features_to_image` = moyenne 8→3 canaux + upscale 64×64.
Ce n'est pas une vraie image.

**Étape 2b — `cnn_invert.py` (gagnant legacy)** :

- Trouver x (3,64,64) tel que `activation(conv(x)) ≈ target`
- `loss = MSE(pred, target) + tv_weight * TV(x)`
- `conv` = vrais poids du state_dict
- Seule approximation : `adaptive_avg_pool` si taille ≠ feature_shape
- Adam, 2000–5000 steps, garde le meilleur feat_loss

`fc1_analytic.py` (ajouté plus tard) : score de confiance d'isolement des lignes fc1 avant
sélection (détecter model6 contaminé).

#### B. MLP (modèles 1,4,5,8)

Extraction : même formule sur net.0 — la ligne est déjà l'image aplatie (12288 = 3×64×64).

```python
# extract_mlp → flat_to_image → dedup_select 128 images
```

Pas de conv à inverser : l'analytique est la reconstruction directe (exacte si un seul neurone
ReLU isolé).

`mlp_reconstruct.py` : pour ReLU, tri par `own_activation` (neurone actif sur sa propre ligne)
+ dedup.

#### C. ViT (modèles 9, 11)

Pas d'analytique propre → `run.py --optimize --allow-guessed-forward`.

- Rebuild `SimpleViT` depuis state_dict (noms timm)
- Labels via iDLG (`infer_labels` sur head.bias)
- `gradient_match` : minimise `1 - cos(grad_simulé, G_observé) + TV(x)`
- **Fix model11** : variante `global_pool=avg` → `fc_norm` après mean-pooling, pas de
  `cls_token`. Ancien code attendait `norm` → rebuild échouait → bruit.

### 5. Chronologie de cette conversation

#### Phase 1 — Comprendre CNN et ablation

- Explication : x, conv(x), pred, target, 3000 itérations
- Ablation par modèle (`inv2.pt` … `inv12.pt`) depuis `submission_analytic.pt`
- `feature_mse` : 2,10,12 excellents ; 3,7 moyens ; 6 ≈ 17 (bloqué)

#### Phase 2 — Consolidation CNN

| Soumission | Score | Contenu |
|---|---|---|
| 192 | 0.2404 | Tous CNN inversés |
| 316 | 0.2418 | `submission_all_m3_5k.pt` (model3 @ 5000) |

#### Phase 3 — Tests d'exclusion CNN (erreur de hypothèse, mais utile)

| Fichier | Score | Leçon |
|---|---|---|
| `submission_no6.pt` | 0.239 | −0.0027 — ne pas enlever model6 |
| `submission_no6_no7_m3_5k.pt` | 0.229 | −0.0128 — ne pas enlever 6 ni 7 |
| `submission_all_m3_m7_5k.pt` | 0.2417 | model7@5k inutile |

**Règle validée :** ne jamais remplacer une inversion CNN par l'analytique, même si MSE local
est catastrophique (model6).

#### Phase 4 — MLP (échec confirmé, legacy)

Diagnostic :

- model1 sigmoid: quality top128 ~0.077, own_active ~0.50
- model4 tanh   : ~0.080, own_active ~0.47
- model5 relu   : ~0.080, own_active ~0.66
- model8 relu   : ~0.065, own_active ~0.57

| Test | Score | vs 0.2418 |
|---|---|---|
| `sub_mlp_sel.pt` (sélection améliorée) | 0.2367 | −0.005 |
| `sub_mlp_refine.pt` (gradient match 1,4) | 0.1997 | −0.042 |

**MLP fermé (legacy) :** garder l'analytique déjà dans la base.

> **Note v2 :** `bench_selection.py` montre que la sélection `own_margin` est quasi-optimale
> en régime isolé ; le refine MLP legacy échouait sans garde anti-régression et sans warm-start
> v2. À re-tester avec `run.py --optimize` si besoin (§8 corps principal).

#### Phase 5 — ViT (succès legacy)

| Étape | Résultat |
|---|---|
| `inspect_models.py --keys 9 11` | Architectures différentes |
| Rebuild test | model9 OK ; model11 FAILED (fc_norm vs norm) |
| `sub_vit.pt` (job avec 9+11, mais 11 en bruit) | 0.2432 (+0.0014) |
| Fix `rebuild.py` SimpleViT | Support fc_norm + avg pool |
| **`sub_vit_both.pt`** | **0.2469 (+0.0037) — record legacy** |

#### Phase 6 — Pipeline v2 analytique (session actuelle)

- Fix couleur CNN (transmit filters), filtre de structure, sélection `own_margin`
- **`submission_v2.pt` → 0.255 (ID 1114)** — nouveau record
- Benchmark synthétique : plafond analytique MLP ~0.42 ; gain restant = GPU sur lisses + ViT

#### Phase 7 — Suite recommandée (à faire)

- Jobs `sub_vit_12k.pt` (9+11, 12000 steps) et `sub_vit9_15k.pt` (model9 seul, 15000 steps)
  — si lancés, vérifier résultats
- Fusionner legacy CNN invert + ViT dans base v2 (0.255)
- GPU MLP lisses 1,4 avec garde anti-régression (`sbatch --array=1,4 slurm_array.sh`)

### 6. Tableau complet des soumissions (conversation)

| ID | Score | Fichier / contenu | Note |
|---|---|---|---|
| 273 | 0.1675 | `inv2.pt` | Ablation model2 seul |
| 286 | 0.1711 | `inv12.pt` | Ablation model12 seul |
| 316 | 0.2418 | `submission_all_m3_5k.pt` | CNN + m3@5k |
| 329 | 0.239 | `submission_no6.pt` | Exclusion 6 ❌ |
| 342 | 0.229 | `submission_no6_no7_m3_5k.pt` | Exclusion 6+7 ❌ |
| 433 | 0.2417 | `submission_all_m3_m7_5k.pt` | m7@5k inutile |
| 524 | 0.2367 | `sub_mlp_sel.pt` | MLP sélection ❌ |
| 542 | 0.1997 | `sub_mlp_refine.pt` | MLP refine ❌ |
| 599 | 0.2432 | `sub_vit.pt` | ViT 9 seul (11 bruit) |
| 645 | 0.2469 | `sub_vit_both.pt` | ★ Best legacy |
| 880 | 0.2033 | v2 avec clamp | Régression (clamp vs to_unit) |
| **1114** | **0.255** | **`submission_v2.pt`** | **★ Best actuel** |

Référence historique (avant conversation) : baseline analytique 0.1427 ; désastre optimize all
0.0635.

### 7. Choix stratégiques — pourquoi

**Principe directeur :** minimiser les hypothèses inventées. Utiliser uniquement ce qui est
exact dans le state_dict + le gradient fourni.

| Choix | Pourquoi |
|---|---|
| CNN : inversion conv seule | conv exacte ; fc1/head/labels = trop d'inconnues → 0.0635 |
| Pas de full gradient match sur CNN | Forward CNN deviné (pool/stride) |
| Garder les 6 inversions CNN | Exclusion mesurée = baisse SSIM |
| MLP : analytique seul (legacy) | Lignes = images ; refine dégrade SSIM (tests legacy) |
| ViT : gradient match sur forward strict | Seule voie ; 9/11 étaient du bruit |
| Base `submission_all_m3_5k.pt` puis `sub_vit_both.pt` | Toujours patcher par-dessus le best |
| `mean_quality` ≠ SSIM | Ne pas décider des soumissions dessus |

**Ce qui a marché**

- `cnn_invert.py` — passage ~0.14 → ~0.24
- model3 @ 5000 steps — +0.0014
- ViT model9 — +0.0014
- Fix model11 + optimize — +0.0037
- Pipeline v2 (transmit + structure gate) — 0.2469 → **0.255**

**Ce qui n'a pas marché**

- Optimize tous modèles (CNN/ViT devinés) → 0.0635
- MLP gradient matching (legacy, sans garde) → 0.0761 / 0.1997
- Exclure CNN « ratés » (no6, no7)
- Sélection MLP « own_active » sur 5,8 (legacy)
- kmeans global (noté plus tôt)
- `clamp(0,1)` au lieu de `to_unit()` → 0.2033
- Overfit sur proxy local vs leaderboard

### 8. Erreurs à ne pas refaire

| Erreur | Conséquence | À faire à la place |
|---|---|---|
| Gradient match sur forward deviné (CNN/ViT cassé) | Score effondré | Vérifier `build_*` strict load avant GPU |
| Remplacer inversion CNN par analytique | −0.003 à −0.013 | Toujours garder les 6 inversions (legacy) |
| Refine MLP 1,4 sans garde (legacy) | −0.005 à −0.042 | `run.py --optimize` avec anti-régression, ou ne pas toucher |
| Croire MSE local = SSIM (model6) | Mauvaise exclusion | Soumettre / ablation serveur |
| Optimiser model7@5k sans gain | Temps GPU perdu | Mesurer avant de généraliser |
| `sub_vit` sans fix model11 | 11 reste bruit | `inspect` + test rebuild d'abord |
| Coller du texte terminal dans bash | syntax error | Une commande à la fois |
| Commit consignes et `soumissions.txt` | Fuite API key | `.env` seulement |
| Spam soumissions (< 5 min) | 429 cooldown | Attendre puis relancer `task_template.py` |
| `clamp(0,1)` normalisation | 0.2033 vs 0.2469 | Toujours `utils.to_unit()` |

### 9. Flux de données (résumé visuel)

```
CLIENT (128 images) → entraînement → gradient MOYEN par paramètre
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                         ▼                         ▼
                 MLP net.0                  CNN fc1                    ViT
            row = gW/gb              row = gW/gb (features)         pas d'analytique
            = image directe          → cnn_invert(conv)            → gradient_match
                    │                  ou v2: transmit invert           │
                    └─────────────┬─────────────┴─────────────────────────┘
                                  ▼
                    submission.pt : model1..model12 × (128,3,64,64)
                                  ▼
                         SSIM leaderboard (matching)
```

### 10. Fichiers de soumission à retenir

| Fichier | Rôle |
|---|---|
| **`submission_v2.pt`** | **Best actuel 0.255 — base de travail v2** |
| `sub_vit_both.pt` | Best legacy 0.2469 — ViT 9+11 optimisés, à fusionner dans v2 |
| `submission_all_m3_5k.pt` | CNN optimisés (sans ViT) — source CNN invert |
| `sub_vit.pt` | Intermédiaire (9 seul) |
| `sub_mlp_*`, `submission_no6*` | Ne pas resoumettre |

**Workflow soumission :**

```bash
python submit.py --check FICHIER.pt
cp FICHIER.pt $TASK3_DATA_ROOT/submission.pt
cd $TASK3_DATA_ROOT && python task_template.py
cd -
```

### 11. Commandes cluster utiles

```bash
cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1
export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction
source $TASK3_DATA_ROOT/.venv/bin/activate
git pull origin main
```

**Diagnostics :**

```bash
python diagnose_mlp.py --models 1 4 5 8
python inspect_models.py --keys 9 11
python analyze.py --fc1-confidence   # CNN contamination
python reconstruct_v2.py --diagnose  # v2 recoverability + previews
```

**GPU :**

```bash
sbatch slurm_cnn_invert.sh   # variables CNN_MODELS, CNN_OUT, CNN_STEPS, CNN_BASE
sbatch slurm_mlp_refine.sh   # déconseillé (pas de garde anti-régression)
sbatch --array=1,4 slurm_array.sh   # v2: MLP lisses avec garde (run.py --optimize)
# ViT : run.py --optimize --allow-guessed-forward --models 9 11 --base sub_vit_both.pt
```

**Exemples CNN invert (legacy, prouvés) :**

```bash
# Affiner model7 depuis le best
sbatch --export=ALL,CNN_MODELS="7",CNN_OUT=submission_all_m3_m7_5k.pt,CNN_STEPS=5000,CNN_BASE=submission_all_m3_5k.pt slurm_cnn_invert.sh

# Rebuild sans model6
sbatch --export=ALL,CNN_MODELS="2 3 7 10 12",CNN_OUT=submission_no6.pt,CNN_STEPS=3000,CNN_BASE=submission_analytic.pt slurm_cnn_invert.sh

# Chain job (no6 + m3 5k) via --wrap
sbatch --account=training2625 --partition=dc-gpu --reservation=cispahack \
  --nodes=1 --gres=gpu:1 --cpus-per-task=8 --time=01:30:00 \
  --job-name=no6_m3_5k --output=output/no6_m3_5k_%j.out \
  --wrap='module purge; export TASK3_DATA_ROOT=/p/scratch/training2625/dougnon1/Loki/FL_Data_Reconstruction; cd /p/scratch/training2625/dougnon1/Loki/cispa_final/task_3_fl_reconstruction/attempt1; source "$TASK3_DATA_ROOT/.venv/bin/activate"; python cnn_invert.py --base submission_analytic.pt --out submission_no6_tmp.pt --models 2 7 10 12 --steps 3000; python cnn_invert.py --base submission_no6_tmp.pt --out submission_no6_m3_5k.pt --models 3 --steps 5000; python submit.py --check submission_no6_m3_5k.pt'
```

### 12. Où est le gain restant (fin de conversation)

| Piste | Potentiel | Risque |
|---|---|---|
| **Fusionner v2 (0.255) + CNN invert legacy + ViT legacy** | **Élevé** | Faible si patch modèle par modèle |
| ViT plus de steps / model9 seul | Moyen (+0.001–0.005) | Faible (base intacte) |
| ViT multi-seed | Moyen | Coût GPU |
| Fix couleur model12 (v2 CPU) | Moyen | Faible |
| CNN re-tune (fc1 confidence, TV/lr) | Faible (plateau) | Faible |
| MLP refine avec garde (v2) | À mesurer | Moyen |
| Denoising cluster (v2, bench d'abord) | Moyen en régime mixte | Moyen |

**Priorité immédiate :**

1. Débloquer scratch (reconnexion login node)
2. Fusionner `sub_vit_both.pt` (ViT 9+11) et CNN invert legacy dans `submission_v2.pt`
3. Résultats jobs `sub_vit_12k.pt` / `sub_vit9_15k.pt` si lancés
4. Fix model12 couleur (CPU)

### 13. Résumé en 10 lignes

1. On reconstruit 1536 images depuis gradients moyens sur 128 images et modèles connus.
2. **Legacy :** CNN analytic fc1 → `cnn_invert` (vraie conv) = saut 0.14 → 0.24 ; ViT gradient
   matching + fix fc_norm model11 → 0.2469.
3. **v2 :** transmit filters + `own_margin` + structure gate → **0.255** (record actuel).
4. MLP analytique suffit en legacy ; refine legacy dégrade ; v2 montre plafond ~0.42 en synthétique.
5. Best actuel = **`submission_v2.pt` (0.255)** ; legacy ViT/CNN à **réintégrer** dans cette base.
6. Ne jamais exclure un CNN inversé ni refaire full optimize all.
7. MSE local ≠ SSIM ; le serveur garde le best → tests sans risque.
8. Code clé legacy : `cnn_invert.py`, `fc1_analytic.py`, `rebuild.SimpleViT`, `run.py`.
9. Code clé v2 : `reconstruct_v2.py`, `channels.py`, `separation.py`, `bench_selection.py`.
10. Suite : fusionner legacy + v2 ; pousser ViT ; fix model12 ; GPU prudemment.
