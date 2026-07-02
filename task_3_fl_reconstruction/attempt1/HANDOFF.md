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
