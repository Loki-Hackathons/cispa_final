# Task 1 — Les 4 schémas de watermark

Guide pédagogique des quatre familles présentes dans le dataset `SprintML/watermark_localization`. Paramètres exacts et clés : [`task_1_text_watermark/watermark_config.yaml`](../../task_1_text_watermark/watermark_config.yaml).

Pour l’implémentation détecteur côté Loki : [`task_1_text_watermark/alexandre/detectors.py`](../../task_1_text_watermark/alexandre/detectors.py) · pipeline : [`attempt1.md`](attempt1.md).

---

## Vue d’ensemble

| Schéma | Type de signal | Dépend du contexte ? | Params hackathon |
|--------|----------------|----------------------|------------------|
| **Unigram** | Binaire (green / red) | Non — greenlist fixe | `fraction=0.5`, `strength=1.0` |
| **KGW** | Binaire | Oui — greenlist par n-gram | `gamma=0.25`, `delta=1.5` |
| **Gumbel-Max** | Continu | Oui — PRF sur n-gram | `ngram=2`, clé secrète |
| **TextSeal** | Continu | Oui — PRF dual-key sur n-gram | `ngram=3`, `α=0.5`, deux clés |

**Rappel Task 1 :** le label indique si le watermarking était **actif** à la génération, pas si un détecteur passe un test statistique. Les signaux bruts ci-dessous sont bruités ; il faut les agréger (fenêtres, HMM, etc.).

---

## 1. Unigram

### Idée

Une **greenlist fixe** : 50 % des tokens du vocabulaire sont « verts », 50 % « rouges », tirés une fois à partir de la clé secrète (`SHA256(watermark_key)` → shuffle). La liste ne change jamais pendant la génération.

### Génération (watermark ON)

Le vendor (`gptwm.GPTWatermarkLogitsWarper`) **ne force pas** uniquement des tokens verts. Il ajoute un biais aux logits **avant** le softmax :

```python
new_logits = scores + strength * green_list_mask
```

- `green_list_mask` vaut **1.0** (vert) ou **0.0** (rouge) par token du vocabulaire.
- `strength=1.0` dans le YAML hackathon → chaque token vert reçoit **+1.0** sur son logit.

**Important :** les logits du LLM ne sont **pas** bornés à [0, 1]. Ce sont des réels libres (souvent entre −5 et +15). Le masque 0/1 × `strength` est un **increment additif**, pas une probabilité. Exemple :

| Token | logit modèle | masque | logit final |
|-------|-------------|--------|-------------|
| « the » (rouge) | 8.2 | 0 | 8.2 |
| « a » (vert) | 6.5 | 1 | 7.5 |

Ensuite softmax → les verts sont **favorisés**, les rouges restent possibles. Si le boost suffit à renverser le top-1, un vert est choisi à la place du rouge dominant.

### Limites

- La greenlist est **aléatoire** : un token très probable peut être rouge → le watermark peut **distordre** le texte pour imposer un vert.
- **Pas de pondération par entropie** (contrairement à TextSeal).
- Entropie faible + token dominant **vert** → signal watermark faible (le +1 ne change presque rien).
- Entropie faible + token dominant **rouge** → distorsion plus forte, signal plus visible.

### Détection (notre pipeline)

Par token généré : **1** s’il appartient à la greenlist fixe, **0** sinon.

| | Attendu |
|---|---------|
| H0 (texte clean) | ~50 % de verts par hasard |
| H1 (watermark actif) | >50 % de verts sur le span |

Signal très bruité token par token → nécessite lissage temporel (fenêtres, HMM).

---

## 2. KGW (Kirchenbauer et al.)

### Idée

Greenlist **contextuelle** : à chaque position, la liste verte dépend des tokens précédents (n-gram), pas d’une liste globale fixe.

### Génération (watermark ON)

1. Prendre le contexte (schéma hackathon : n-gram de **4 tokens** incluant le target, `self_salt=True`).
2. PRF(contexte, clé) → seed.
3. `torch.randperm(vocab_size)` sur **GPU** (Philox) → **25 %** du vocab = greenlist pour ce contexte (`gamma=0.25`).
4. Ajouter **`delta=1.5`** aux logits des tokens verts (même principe qu’Unigram : biais additif, pas contrainte dure).
5. Softmax + sampling habituel.

Deux occurrences du même mot à des endroits différents → greenlists différentes.

### Gotcha hackathon

Les greenlists du dataset ont été générées avec `torch.randperm` sur un **`torch.Generator` CUDA**. Recalculer sur CPU produit des listes incohérentes → ~1/3 des tokens KGW watermarked semblent clean. **Obligatoire :** `kgw_scores.py` sur JURECA.

### Détection (notre pipeline)

Pour le token à la position `t` : est-il dans **sa** greenlist contextuelle ? → **1** ou **0**.

| | Attendu |
|---|---------|
| H0 (clean) | ~25 % de verts par hasard |
| H1 (watermark actif) | nettement >25 % sur le span |

Plus fin qu’Unigram (contexte), mais toujours binaire et bruité → mêmes besoins d’agrégation.

### Unigram vs KGW en une phrase

- **Unigram :** « Ce token est-il dans la liste verte **globale** ? »
- **KGW :** « Ce token est-il dans la liste verte **pour ce contexte** ? »

---

## 3. Gumbel-Max (Aaronson & Kirchner)

### Idée

Watermark **distortion-free** (en théorie) : on ne biaise pas les logits. On modifie la règle de sampling avec des nombres pseudo-aléatoires dérivés du contexte.

### Génération (watermark ON)

1. Calculer les probabilités du modèle `p(v)` (softmax des logits, éventuellement top-p).
2. Pour **chaque** token candidat `v`, PRF(n-gram contexte, clé secrète) → `r_v ∈ [0,1]` uniforme.
3. Choisir le token : **argmax_v `r_v^(1/p_v)`** (équivalent Gumbel-max : tirage watermarké qui respecte la distribution du modèle).

Param hackathon : n-gram de **2 tokens** de contexte.

### Détection (notre pipeline)

Pour le token **effectivement généré** à la position `t` :

1. PRF sur le n-gram de contexte + token cible → `r`.
2. Score = **`-log(1 - r)`** (increment Gumbel ; sous H0 ~ Exp(1), mean=1, var=1).

Positions `t < ngram` : pas assez de contexte → score = moyenne H0 (1.0).

| | Attendu |
|---|---------|
| H0 (clean) | mean ≈ 1, var ≈ 1 |
| H1 (watermark actif) | scores légèrement plus élevés en moyenne sur le span |

Signal **continu** (pas binaire) ; séparation token à token modeste mais réelle.

---

## 4. TextSeal

### Idée

Extension du Gumbel-Max avec **deux clés secrètes** et un mélange aléatoire (`α`) pour restaurer la diversité des sorties tout en restant détectable. Watermark localisable dans des documents dilués (papier TextSeal).

### Génération (watermark ON)

1. Probabilités du modèle + top-p (comme Gumbel-Max).
2. PRF dual-key : pour chaque candidat, tirer `r_a` (clé A) et `r_b` (clé B).
3. À chaque step, choisir aléatoirement la clé A avec probabilité **`α=0.5`**, sinon clé B.
4. Sampling Gumbel-max avec le `r` de la clé active : **argmax `log(r) / p`** (formulation TextSeal dans le vendor).

Param hackathon : n-gram de **3 tokens**, `key_a`, `key_b`, `mixing_alpha=0.5`.

### Détection (notre pipeline)

Pour le token généré :

1. `r_a = PRF(contexte, token, key_a)`, `r_b = PRF(contexte, token, key_b)`.
2. Score fusionné : **`α · g(r_a) + (1-α) · g(r_b)`** avec `g(r) = -log(1-r)`.

| | Attendu |
|---|---------|
| H0 (clean) | mean ≈ 1, var ≈ 0.5 |
| H1 (watermark actif) | mean ≈ 1.1–1.2 sur les spans (séparation modeste token à token) |

Signal continu. Le détecteur officiel TextSeal utilise aussi pondération par **entropie** et recherche de segments (geometric cover) — notre pipeline Task 1 ne fait **que** le score PRF token-level (pas de forward pass LLM).

### Gumbel-Max vs TextSeal

| | Gumbel-Max | TextSeal |
|---|------------|----------|
| Clés | 1 | 2 (mix α=0.5) |
| N-gram | 2 | 3 |
| Variance H0 | 1.0 | 0.5 |
| Diversité | une seule clé | dual-key routing |

---

## Logits vs scores de soumission

Ne pas confondre :

| Objet | Espace | Rôle |
|-------|--------|------|
| **Logits LLM** | ℝ (non bornés) | Sortie du modèle avant softmax ; Unigram/KGW y ajoutent un boost |
| **Signaux détecteur** | ℝ ou {0,1} | Statistique par token (PRF, greenlist) — entrée de notre pipeline |
| **Scores soumis Task 1** | [0, 1] | Confiance « watermark actif » — produits par fusion/HMM, pas par les watermarks eux-mêmes |

---

## Références

- KGW : Kirchenbauer et al., ICML 2023 — [`lm-watermarking`](../../task_1_text_watermark/vendor/lm-watermarking/)
- Gumbel-Max / TextSeal : repo [`textseal`](../../task_1_text_watermark/vendor/textseal/) (commit épinglé dans le YAML)
- Unigram : Zhao et al., ICLR 2024 — [`unigram-watermark`](../../task_1_text_watermark/vendor/unigram-watermark/)
- TextSeal (papier assigné) : [`docs/TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md`](../TextSeal_a_localized_llm_watermark_for_provenance_and_distillation_protection.md)
