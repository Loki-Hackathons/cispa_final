# Task 2 — MGI : roadmap ultra-détaillé (attempt1)

**Rédigé :** 2026-07-03. **Owner :** Florian (dougnon1). **Spec :** [Task 2 Description.md](../Task%202%20Description.md) · **Paper :** [mgi-member-vs-generated-inference.md](../mgi-member-vs-generated-inference.md)

**Métrique :** `score = mean_{i=1..1800}[ DetectorScore_i × (1 − MSE_i) ]`, `DetectorScore ∈ {0,1}` (1 si le flip ciblé réussit), `MSE` = MSE normalisée (pixels [0,1]) entre l'image soumise et **son image de référence source**. Borne [0,1]. Public = 30 % / privé = 70 %.

**Règle d'or :** `DetectorScore` est binaire et **domine** `(1−MSE)`. Un flip qui réussit à MSE=0.1 vaut 0.9 ; un flip raté à MSE=0.0001 vaut 0.0. **La priorité absolue est de faire flipper (transfert vers le détecteur caché), la MSE est secondaire.** On n'optimise la MSE qu'une fois le flip fiable.

---

## 0. Modèle de menace — ce qu'est (très probablement) le détecteur caché

La spec nous donne **RAR** (Randomized Autoregressive model, `yucornetto/RAR`) et cite explicitement le papier **DCB** (Data Circuit Breaker). Le détecteur caché est, avec forte probabilité, **DCB en réglage *direct training* (M₁)** sur RAR :

| Étage | Signal | Rôle | Nous l'avons ? |
|-------|--------|------|----------------|
| **Stage 1** | `L_A = L_R + α·L_Q` (autoencodeur MaskGIT VQ-GAN) | sépare **G** (généré) du naturel. Généré ⇒ `L_A` **bas**. | ✅ `proxy_dcb.py` |
| **Stage 2** | ICAS = `nll_uncond − nll_cond` (générateur latent RAR) | sur le non-généré, sépare **M** de **N**. Membre ⇒ ICAS **haut**. | ✅ `proxy_icas.py` |
| **Stage 3** | cross-generator KDE | seulement pour le réglage *derivative* (M₂). | ❌ non pertinent ici |

**Le point stratégique n°1 :** le tokenizer de RAR **EST** le MaskGIT VQ-GAN (`maskgit-vqgan-imagenet-f16-256`). Notre proxy Stage-1 n'est donc **pas un modèle de substitution approximatif — c'est le composant exact du détecteur.** L'attaque adverse sur `L_A` doit donc transférer **quasi parfaitement** (à la marge de seuil près). Idem Stage 2 si le détecteur utilise RAR. **Conséquence : l'attaque adverse à basse MSE peut battre le content-swap à haute MSE — à condition de valider le transfert.**

**Ordre des scores (établi par le diagnostic `score_membership --diagnose`) :** `ICAS(N) < ICAS(M) ≲ 0.205 < ICAS(G)`. Généré = ICAS le plus haut ; non-membre = le plus bas. La séparation M/N est réelle (TPR .957 / FPR .043 sur seuil médian).

### 0.1 Carte des 6 directions × difficulté (le cœur du plan)

Chaque cellule cible est un couple **(Stage 1, Stage 2)**. Réf. = image source du slot.

| Dir | Slots | Réf source | Cible Stage 1 (`L_A`) | Cible Stage 2 (ICAS) | Recette | Difficulté |
|-----|-------|-----------|----------------------|----------------------|---------|------------|
| **M→G** | 0300–0599 | M (000–299) | **généré** : baisser `L_A < τ_G` | — (Stage 1 tranche) | reconstruction VQ-GAN | **facile** |
| **N→G** | 0900–1199 | N (300–599) | **généré** : baisser `L_A` | — | reconstruction VQ-GAN | **facile** |
| **G→M** | 1200–1499 | G (600–899) | **naturel** : monter `L_A > τ_G` | membre : ICAS déjà haut ⇒ garder | remonter `L_A` seul | **facile-moyen** |
| **M→N** | 0000–0299 | M (000–299) | naturel (déjà) : garder `L_A` haut | non-membre : **baisser** ICAS sous frontière M/N | attaque ICAS ↓ + garde `L_A` | **moyen** |
| **G→N** | 1500–1799 | G (600–899) | **naturel** : monter `L_A` | non-membre : **baisser** ICAS | **double objectif** `L_A`↑ ET ICAS↓ | **difficile** |
| **N→M** | 0600–0899 | N (300–599) | naturel (déjà) : garder `L_A` | membre : **monter** ICAS au-delà du max des vrais membres | fabriquer ICAS >> 0.205, garder `L_A` naturel | **difficile** |

**Insight clé (note de `attack_membership.py`) :** soumettre le vrai membre le plus mémorisé ne flippe PAS →M (le seuil membre du détecteur est **au-dessus** de tous les membres réels). Pour →M il faut **fabriquer** une image dont l'ICAS dépasse la plage naturelle, sans faire chuter `L_A` (sinon → classé G). C'est pour ça que N→M et G→N sont les deux directions dures.

---

## 1. Ce qui est cassé / manquant aujourd'hui (audit code)

| # | Problème | Fichier | Impact | Priorité |
|---|----------|---------|--------|----------|
| 1 | **`M_N` et `N_M` = `"original"`** dans `DIRECTION_BLOCKS` → soumises inchangées | `config.py` | **2/6 directions scorent 0** dans le pipeline principal | **P0** |
| 2 | **`G_M` et `G_N` = même objectif `from_G`** | `config.py`/`cw_attack.py` | G→N ne contrôle jamais Stage 2 → probablement classé M ⇒ **G→N score 0** | **P0** |
| 3 | **Stage 2 (ICAS) jamais intégré** au pipeline principal | `run_attack.py` | aucune direction M/N pilotée dans le flux `--phase all` | **P0** |
| 4 | **Aucun scoreur local proxy-DCB** | — | on ne peut pas mesurer le flip avant l'API ⇒ itération à l'aveugle | **P0** |
| 5 | **Marge de seuil `κ=0.05` fixe** : on traverse *notre* `τ_G` de justesse | `cw_attack.py` | si le seuil réel diffère, le flip ne transfère pas | **P1** |
| 6 | **JPEG q=80 forcé sur toutes les images** | `submission_io.py` | dégrade la MSE et peut casser les flips adverses limites ; jamais comparé à LZMA brut | **P1** |
| 7 | **Classe ImageNet devinée par ResNet-50** pour l'ICAS | `proxy_icas.py` | conditionnement RAR faux ⇒ ICAS mal calibré (surtout →M / →N-parmi-générés) | **P1** |
| 8 | **Réplication manuelle du forward RAR** (fragile) | `attack_membership.py` | correcte mais non couverte hors `--selftest` ; à re-valider à chaque màj de poids | **P1** |
| 9 | **Deux familles de pipeline disjointes** (`run_attack` C&W vs `build_submission_v2` blocs) | — | pas de source de vérité unique ; risque d'incohérence des soumissions | **P1** |
| 10 | **Aucune trace leaderboard** (0 soumission loggée) | `history/` | on ne sait pas ce qui transfère réellement | **P0 (process)** |
| 11 | **`from_G` ne pilote pas la classe M/N** post-dé-génération | `cw_attack.py` | racine de #2 | **P0** |

---

## 2. Workstream A (P0) — Scoreur local proxy-DCB **[FAIT : `evaluate_submission.py`]**

**Le multiplicateur de force n°1.** Sans détecteur, on reconstruit **notre** DCB (Stage 1 `L_A` seuillé à `τ_G` + Stage 2 ICAS seuillé à `τ_MN`) et on score n'importe quel `submission.npz` :

- par slot : `L_A`, ICAS → classe prédite ∈ {M,N,G} ; flip = (classe == cible) ; `MSE` vs réf source ;
- par direction : **flip-rate**, MSE moyenne, score estimé `mean(flip×(1−MSE))` ;
- global : score estimé total + décomposition par direction.

**Pourquoi c'est P0 :** transforme chaque itération d'attaque d'un « pari API » (cooldown 5 min, feedback binaire) en une **mesure locale immédiate**. Toute la suite se règle dessus.

**Limite honnête :** mesure le transfert vers *notre* proxy, pas vers le détecteur caché. Mais comme le proxy Stage-1 = composant exact du détecteur, un flip proxy Stage-1 est un **très fort** prédicteur. Stage-2 dépend de l'exactitude du conditionnement de classe (cf. Workstream E).

**Go/no-go :** ne soumettre à l'API que si `evaluate_submission.py` donne un flip-rate proxy ≥ 0.9 sur la direction concernée. Sinon, itérer localement.

```bash
# sur nœud GPU (Stage 2 = RAR requis) ; Stage 1 seul possible sans RAR
python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz
python task_2_mgi/attempt1/evaluate_submission.py output/submission.npz --stage1-only
```

---

## 3. Workstream B (P0) — Attaque unifiée par cellule cible **[FAIT : `attack_combined.py`]**

Remplace les objectifs partiels par **une** attaque qui vise la cellule (Stage 1, Stage 2) exacte de chaque direction, avec pertes combinées :

- perte Stage 1 : `hinge(L_A − (τ_G−κ))` pour « générer », `hinge((τ_G+κ) − L_A)` pour « naturaliser » ;
- perte Stage 2 : `hinge(ICAS_cible − ICAS)` (monter) ou `hinge(ICAS − ICAS_cible)` (baisser), différentiable via le chemin soft-token straight-through **déjà validé** dans `attack_membership.py` ;
- terme MSE `c_mse·‖x−x₀‖²` ;
- **validation « deployable »** : succès mesuré sur l'uint8 (+ JPEG optionnel) re-encodé, jamais sur le float32 (évite le faux positif de gradient).

Directions & objectifs (résout #2, #3, #11) :

| Dir | Stage 1 | Stage 2 |
|-----|---------|---------|
| M→G, N→G | pousser généré | off |
| G→M | naturaliser | garder ICAS haut (déjà) / pousser si besoin |
| M→N | garder naturel | baisser ICAS < τ_MN |
| G→N | naturaliser | baisser ICAS < τ_MN |
| N→M | garder naturel | monter ICAS > `ICAS_max(M) + marge` |

**Prérequis :** `--selftest` (réplication forward == vérité terrain) DOIT passer avant tout run réel. Puis valider chaque bloc avec Workstream A.

**Go/no-go :** un bloc n'entre en soumission que si (selftest PASS) ET (flip-rate proxy ≥ 0.9). Pour N→M et G→N, si le flip-rate plafonne < 0.5 malgré la montée d'ICAS, **fallback content-swap** (cf. Workstream C).

---

## 4. Workstream C (P0, filet de sécurité) — recettes non-adverses par direction

Toujours garder une **soumission de repli qui score** même si l'adverse ne transfère pas :

- **`*→G` :** `reconstruct_to_g.py` (encode-decode VQ-GAN). Estampille le fingerprint décodeur exact, MSE basse. **Meilleure recette facile.**
- **`G→M` :** remonter `L_A` d'un généré (l'ICAS est déjà haut) — souvent suffit pour membre.
- **`→M` / `→N` durs :** `build_content_swap.py` (plus proche voisin de la classe cible). MSE ~0.05–0.15 mais flip robuste (vraie image de la classe). C'est le **plancher garanti** pour N→M et G→N.
- **`M→N` :** perturbation minime (noise/jpeg/shift via `build_submission_v2`) casse souvent la mémorisation à MSE quasi nulle — à comparer en Workstream A à l'attaque ICAS.

**Stratégie de composition (le vrai levier de score) :** les 6 directions sont **indépendantes**. `build_submission_v2 --dir NAME=SPEC` permet de choisir, **par direction**, la méthode qui donne le meilleur `flip×(1−MSE)` mesuré (proxy A, puis confirmé API). On construit la soumission finale en cueillant le meilleur par direction.

---

## 5. Workstream D (P1) — MSE & format de soumission

1. **Comparer LZMA brut vs JPEG** : `submission_io.save_submission_npz` force q=80. Mesurer la taille LZMA sans JPEG (1800×256²×3 = 354 Mo bruts, mais content-swap = images naturelles qui compressent bien). Si < 200 Mo sans JPEG → **supprimer le JPEG** (gain MSE direct + ne casse plus les flips limites). Sinon, JPEG **seulement** sur les blocs qui le tolèrent (swaps), pas sur les blocs adverses basse-MSE.
2. **Budget MSE par direction :** viser reconstruction (MSE ~1e-3) pour `*→G`/`G→*`, perturbation minimale pour `M→N`, swap plus-proche-voisin pour N→M/G→N. `1−MSE` attendu : facile ≥ 0.99, dur ≥ 0.85.
3. **Marge de seuil adaptative** (résout #5) : au lieu de `κ` fixe, pousser `L_A`/ICAS **franchement** dans le régime cible (p.ex. viser le quantile 10 % de la distribution G pour « générer », pas juste `τ_G−κ`). Coûte un peu de MSE, achète du transfert. À arbitrer en A.

---

## 6. Workstream E (P1) — Fiabiliser le conditionnement de classe (Stage 2)

L'ICAS dépend de la classe conditionnante. Aujourd'hui : ResNet-50 top-1 (≈80 % correct). Améliorations :

- **Classe = argmax de vraisemblance RAR** : pour chaque image, prendre la classe `c` qui minimise `nll_cond` parmi le top-k ResNet (le détecteur, s'il devine la classe, fera pareil). Rend l'ICAS auto-cohérent avec le générateur.
- **Vérifier la sensibilité** : recalculer les distributions M/N/G d'ICAS avec (a) ResNet top-1, (b) argmax-vraisemblance. Si la séparation M/N s'améliore → adopter.
- Documenté dans le paper (§6.7) : l'estimation de prompt dégrade peu DCB — mais ici on **attaque**, donc un conditionnement faux nous fait optimiser la mauvaise cible.

---

## 7. Workstream F (P2) — robustesse & durcissement

- **Multi-α / multi-τ ensemble** : le détecteur peut utiliser un `α`/`τ_G` différents. Optimiser `L_A` pour **plusieurs** `α` simultanément (moyenne des pertes) rend le flip robuste au choix de calibration (analogue à l'EOT).
- **Robustesse aux transforms** (paper §6.5 : JPEG/resize/saturation) : le détecteur peut pré-traiter. Ajouter ces augmentations dans la boucle d'attaque (EOT) pour que le flip survive à un pipeline web.
- **`input_diversity` déjà présent** dans `cw_attack` — l'étendre à `attack_combined`.

---

## 8. Séquencement proposé

| Étape | Action | Où |
|-------|--------|-----|
| T+0 | **A** : lancer `evaluate_submission.py` sur la baseline actuelle (mesure le point de départ réel) | GPU |
| T+0 | **C** : régénérer les blocs faciles `reconstruct_to_g` (M→G, N→G) + swap (N→M, G→N) comme **filet** | GPU |
| T+30 | **A** : scorer la soumission filet → 1re soumission API de référence (logguer `--method`) | GPU+API |
| T+1h | **B** : `attack_combined --selftest` puis M→N, G→M (faciles-moyens) → scorer en A | GPU |
| T+2h | **B** : N→M et G→N (durs) → scorer en A ; si flip < 0.5 garder le swap | GPU |
| T+3h | **D** : arbitrer JPEG vs LZMA ; marges de seuil ; composition best-par-direction | GPU |
| T+4h | **E** : conditionnement argmax-vraisemblance si Stage-2 plafonne | GPU |
| Fin | **Figer** : vérifier format (1800 slots, shape 256×256×3, chaque ID une fois, < 200 Mo), soumettre la meilleure combinaison | API |

---

## 9. Estimations de gain (honnêtes)

| Direction | Recette recommandée | Flip attendu | (1−MSE) attendu | Contribution /6 |
|-----------|--------------------|--------------|-----------------|-----------------|
| M→G | reconstruction | 0.9–1.0 | ~0.99 | forte |
| N→G | reconstruction | 0.9–1.0 | ~0.99 | forte |
| G→M | `L_A`↑ (adverse) ou swap | 0.7–1.0 | 0.9–0.99 | forte |
| M→N | ICAS↓ (adverse) ou perturbation | 0.6–0.9 | ~0.99 | moyenne |
| N→M | swap (filet) / ICAS↑ (upside) | 0.5–0.9 (swap) | 0.85–0.9 | moyenne |
| G→N | swap (filet) / double-obj (upside) | 0.5–0.9 (swap) | 0.85–0.9 | moyenne |

**Chemin réaliste :** score public **~0.55–0.75** (4 directions fortes + 2 filets swap). **Upside** si l'adverse basse-MSE transfère sur N→M/G→N : **0.85+**. **Plancher** (tout en swap qui flippe) : ~0.55–0.65 selon la MSE des swaps.

---

## 10. Ce qu'il ne faut PAS faire

| Piste | Pourquoi |
|-------|----------|
| Soumettre `M_N`/`N_M` inchangés (état actuel `run_attack --phase all`) | score 0 garanti — bug #1 |
| Utiliser `from_G` identique pour G→M et G→N | G→N ne contrôle pas Stage 2 — bug #2 |
| Optimiser la MSE avant d'avoir un flip fiable | `DetectorScore` domine ; MSE inutile si flip=0 |
| Traverser `τ_G` de justesse (`κ` minuscule) | ne transfère pas si le seuil réel diffère — pousser franc |
| Soumettre sans passer par `evaluate_submission.py` | brûle le cooldown API à l'aveugle |
| Faire confiance à `attack_combined` sans `--selftest` PASS | la réplication forward RAR est fragile |

---

## 11. Fichiers (attempt1) — rôle après refonte

| Fichier | Rôle | État |
|---------|------|------|
| `config.py` | chemins, hyperparams, layout | **corrigé** : cellules cibles (Stage1,Stage2) par direction |
| `proxy_dcb.py` | `L_Q, L_R, L_A`, calibration `τ_G/α` | inchangé (correct) |
| `proxy_icas.py` | ICAS RAR (Stage 2) | inchangé (+ option argmax-vraisemblance en E) |
| `cw_attack.py` | C&W L2 sur `L_A` (Stage 1) | conservé pour `*→G` |
| `attack_membership.py` | forward RAR différentiable + selftest | **socle réutilisé** par `attack_combined` |
| `attack_combined.py` | **attaque unifiée par cellule** | **NOUVEAU** |
| `evaluate_submission.py` | **scoreur proxy-DCB local** | **NOUVEAU** |
| `reconstruct_to_g.py` | blocs `*→G` (filet) | conservé |
| `build_content_swap.py` | swap plus-proche-voisin (filet →M/→N) | conservé |
| `build_submission_v2.py` | composition best-par-direction | conservé (source de vérité de l'assemblage) |
| `run_attack.py` | CLI legacy Stage-1 | **déprécié** au profit de `attack_combined` + `build_submission_v2` |
