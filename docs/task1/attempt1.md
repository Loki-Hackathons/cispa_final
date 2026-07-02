# Tâche 1 — Attempt 1 (Alexandre)

**Responsable :** Alexandre (ansart1)
**Code :** [`task_1_text_watermark/alexandre/`](../../task_1_text_watermark/alexandre/)
**Spec :** [`docs/subject/task_1.md`](../subject/task_1.md)

---

## État actuel (2026-07-02, fin de journée)

| Élément                                       | Statut                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| Pipeline HMM (v1, abandonné)                   | ❌ remplacé — trop de faux positifs de bordure, score plafonné            |
| Pipeline semi-Markov (v2, **retenu**)          | ✅ implémenté, testé, soumis                                              |
| Soumission #30 (HMM, sans KGW)                 | score public **0.0065**                                                  |
| Soumission #101 (HMM + KGW)                    | score public **0.0071**                                                  |
| Soumission #159 (semi-Markov, sans KGW)        | score public **0.2001** — ×28 vs #101                                    |
| **Soumission #262 (semi-Markov + KGW)**        | **score public 0.2029** — nouveau best                                   |
| Val locale TPR@0.1%FPR — HMM v1                | 0.0185 (float) / 0.0140 (arrondi 6 déc.)                                  |
| Val locale TPR@0.1%FPR — semi-Markov v2 (sans KGW) | 0.2822                                                                |
| Val locale TPR@0.1%FPR — semi-Markov v2 (+ KGW)    | **0.3135**                                                            |
| Unigram                                        | ✅ **retrouvé et intégré** — `vocab=152064` confirmé par scan non supervisé sur test (§9) ; gain val nul, gain test attendu faible |

**Meilleur score leaderboard public : 0.2029** (soumission #262, `submission_id=262`). Le gros du gain vient d'un changement de modèle (HMM → semi-Markov à longueurs discrètes) et de la correction d'un bug de signal (n-grammes répétés) ; KGW ajoute un gain propre mais nettement plus modeste sur test que sur val (voir §8).

---

## 1. Objectif et métrique

Pour **chaque token** de chaque document test, prédire un score ∈ [0, 1] : confiance que le token a été généré **pendant qu'un watermarking était actif** (label binaire 0/1 en train/val, caché en test).

La métrique est **TPR @ 0.1% FPR**, calculée sur **tous les tokens poolés** (pas par document) : on fixe le seuil τ tel que ≤ 0.1 % des tokens clean (label 0) aient un score ≥ τ, puis on mesure la fraction de tokens watermarkés au-dessus de ce même τ. C'est une métrique de **rang pur** : le budget de faux positifs autorisé est minuscule (51 tokens clean sur 90 documents de validation), donc les quelques tokens clean les mieux notés déterminent tout le score.

**Quatre familles** de watermark possibles, souvent mélangées dans un même document : TextSeal, Gumbel-Max, Unigram, KGW. On ne prédit pas *quel* schéma — seulement *watermark actif ou non*.

---

## 2. Pourquoi le HMM (v1) a été abandonné

Le premier pipeline (HMM à états `clean` / `schéma i`, transitions fixes, forward-backward) donnait un score val de 0.0185 — cohérent mais très en dessous de ce que les signaux bruts suggéraient. Analyse post-soumission #101 :

- **Score saturation / bleeding de bordure** : le posterior HMM se répand sur les tokens clean adjacents à une vraie span watermarkée. Sur les 51 faux positifs consommant tout le budget FPR à 0.1 %, 31 étaient à moins de 10 tokens d'une frontière de span réelle.
- **KGW dégradait le score val** (0.0185 → 0.0066) au lieu de l'améliorer — signe d'un problème de calibration plus profond que la simple fusion de signaux.
- Un test de log-odds non bornés (sans compression logistique) donnait un résultat identique : le problème n'était pas la précision numérique, mais la structure même du modèle.

**Diagnostic racine, trouvé en creusant plus loin :** un bug de calcul de signal, pas seulement un problème de modèle (voir §4).

---

## 3. Infrastructure

| Ressource                        | Chemin                                                                                             |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Dataset (local)                  | `cispa_final/data/watermark_localization/{train,validation,test}.jsonl`                            |
| Dataset (cluster)                | `/p/scratch/training2625/ansart1/loki/watermark_localization/`                                     |
| Clés + params détecteurs         | [`task_1_text_watermark/watermark_config.yaml`](../../task_1_text_watermark/watermark_config.yaml) |
| Repos vendor (commits épinglés)  | `task_1_text_watermark/vendor/{textseal,lm-watermarking,unigram-watermark}/`                       |
| API                              | `CISPA_BASE_URL` + `CISPA_API_KEY` dans `.env` · task id `30-watermark-localization`                |
| Soumission / historique          | `shared/submit.py` → `history/submissions.jsonl`                                                    |
| Eval locale + viewer             | `shared/task1_eval.py` → `history/task1_viz/*.json`                                                 |

**Tokenizer :** `Qwen/Qwen2.5-7B-Instruct`. On utilise les `token_ids` du JSONL tels quels — pas de retokenization du champ `text`.

**Volumes :** train 90 docs · val 90 docs · test 1 320 docs (~1,4 M tokens).

---

## 4. Bug trouvé : évidence fantôme sur n-grammes répétés

Les PRF (TextSeal, Gumbel-Max, KGW) seedent leur tirage aléatoire sur un n-gramme de contexte. **Si le même n-gramme apparaît deux fois dans un document, le PRF ré-émet exactement le même tirage** — ce n'est pas une nouvelle observation indépendante, juste une répétition.

Un document de validation contenait la séquence répétée `letters . letters . letters . ...` (40+ fois) dans une zone **clean**. Chaque répétition du n-gramme accumulait la même valeur de signal, faisant grimper le z-score de fenêtre à **34** dans une région sans watermark — ce document à lui seul absorbait la moitié du budget de faux positifs à 0.1 % FPR.

**Fix (`detectors.py::_dedup_mask`)** : pour chaque n-gramme de contexte, seule sa **première occurrence** dans le document porte le signal ; les répétitions reçoivent la valeur H0 neutre (aucune évidence). Ce comportement correspond à ce que fait un détecteur correct (déduplication de n-grammes), pas une heuristique ad hoc.

Impact isolé de ce fix (mesuré sur le nouveau modèle) : élimine la quasi-totalité des faux positifs "de répétition" qui dominaient le budget FPR.

---

## 5. Signaux par token (`detectors.py`)

| Schéma         | Signal                                                       | Clés / params (YAML)                                             | H0 (attendu)        |
| -------------- | -------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------- |
| **TextSeal**   | Dual-key Gumbel : `α·g(r_a)+(1-α)·g(r_b)`, `g(r)=-log(1-r)`   | `key_a`, `key_b`, `ngram=3`, `α=0.5`                                | μ=1, σ²=0.5          |
| **Gumbel-Max** | `-log(1-r)` sur PRF uniforme                                  | `secret_key`, `ngram=2`                                             | μ=1, σ²=1 (Exp(1))   |
| **Unigram**    | 1 si token ∈ greenlist fixe                                    | `watermark_key`, `fraction=0.5`, **`vocab_size=152064`** (déduit, §9) | Bernoulli(0.5)         |
| **KGW**        | *pas calculé en local* — nécessite CUDA Philox                | `gamma=0.25`, schéma `ff-anchored_minhash_prf-4-True-1306382177`   | Bernoulli(0.25)        |

**Implémentation PRF :** chargement direct de `vendor/textseal/textseal/watermarking/core.py` (`prf_uniform`) via `importlib`, sans importer le package `textseal` complet (évite la dépendance `nltk`).

**Unigram — retrouvé et intégré** (après deux faux départs). `vocab_size=152064` — la seule valeur compatible avec la génération (masque additionné aux logits du modèle). Signal faible (`strength=1.0`) mais confirmé contre un null de 50 clés leurres sur le test set. Investigation complète en §9. Dédup par **token unique** (pas n-gramme : la greenlist est context-free).

---

## 6. Modèle retenu : semi-Markov à segments (`smm_scorer.py`)

### Pourquoi semi-Markov et pas HMM

Les longueurs de spans observées sur train+val sont **quasi discrètes** :

```
31: 64 spans   47: 133 spans   63: 163 spans
95: 145 spans  159: 158 spans  320: 73 spans   (93% du total, reste = longue traîne)
```

Un HMM standard (transitions token-à-token) ne peut pas exploiter cette structure — il "devine" les frontières une transition à la fois, ce qui produit le bleeding observé en v1. Un modèle **semi-Markov** (segments explicites) score directement des fenêtres de longueur candidate et fait un forward-backward sur les *segmentations* du document plutôt que sur les *tokens*.

### Structure du modèle

- Un document = concaténation de segments : soit un **token clean isolé** (coût nul), soit une **span watermarkée** de longueur `L ∈ LENGTHS` avec un prior de longueur empirique `p(L)` (calibré sur les comptes train+val ci-dessus, + 7% de masse résiduelle pour les longueurs hors-liste et les spans tronquées en bord de document).
- **Émission d'une span** : modèle de décalage gaussien sur les signaux standardisés (z-scores sous H0). Pour un décalage hypothétique `shift`, la log-vraisemblance est `LLR(s, e, schéma) = shift · Σx[s:e] − (e−s) · shift² / 2`, combinée par `logsumexp` sur 3 hypothèses de `shift` par schéma (TextSeal ≈ 0.45–0.9, Gumbel-Max ≈ 0.55–1.1, KGW ≈ 0.6–1.3 — toutes estimées/sweepées sur train+val) et sur les schémas disponibles (TextSeal, Gumbel-Max, KGW quand le `.npz` est fourni).
- **Spans tronquées** : gérées explicitement par des hypothèses préfixe `[0, e)` et suffixe `[s, n)` avec un prior dédié, pour les documents qui commencent ou finissent en plein milieu d'un watermark.
- **Score par token** : posterior exact `P(token appartient à un segment watermarké | document)`, calculé par forward-backward en log-espace (`scipy.special.logsumexp`), converti en log-odds puis compressé par une sigmoïde à température (température = 40, choisie pour garder une résolution fine sans jamais saturer à 0/1 exactement).

### Pourquoi ça résout le boundary bleed sans heuristique manuelle

Un token profond dans une span appartient à très peu de segmentations alternatives plausibles (toutes le couvrent). Un token de bordure appartient à des segmentations où il est "dans" la span *et* à des segmentations où il n'y est pas (span plus courte) — le posterior fait cette moyenne automatiquement, pondérée par le prior de longueur. C'est ce que l'heuristique `min(gauche, droite)` visait à approximer manuellement (voir §7) ; le calcul exact fait mieux.

### Résultat

| Version                          | Val TPR@0.1%FPR | Score public |
| ---------------------------------- | ------------------ | -------------- |
| HMM v1 (sans KGW, arrondi 6 déc.) | 0.0140              | 0.0065 (#30)   |
| HMM v1 + KGW                      | 0.0066 ⚠️          | 0.0071 (#101)  |
| Semi-Markov v2 (sans KGW, précision pleine) | 0.2822    | 0.2001 (#159)  |
| **Semi-Markov v2 + KGW**          | **0.3135**          | **0.2029 (#262)** |

---

## 7. Approches intermédiaires testées et abandonnées

| Approche                                         | Fichier              | Val TPR@0.1%FPR | Pourquoi abandonnée                                                                 |
| --------------------------------------------------- | ---------------------- | ------------------ | ---------------------------------------------------------------------------------- |
| Fusion z-score + sigmoïde                          | `build_scores.py`     | ~0.0016             | Pas de structure de span, seuil arbitraire                                          |
| Régression logistique multi-features               | `train_calibrator.py` | ~0.0070             | Classification token-à-token, pas de transitions/segments                          |
| HMM forward-backward                                | `hmm_scorer.py`        | 0.0185              | Bleeding de bordure token-à-token (voir §2)                                        |
| Matched filter (fenêtres discrètes) + `min(g,d)`   | `matched_filter.py`    | ~0.023              | Sélection gloutonne de fenêtres non-chevauchantes = perte d'info vs posterior exact |

Conservées dans le repo comme référence ; **ne pas les réutiliser pour une soumission**.

---

## 8. Intégration KGW

KGW nécessite `torch.randperm` sur un `torch.Generator` **CUDA** (Philox) pour reproduire exactement la génération des organisateurs — impossible à recalculer sur CPU (sur CPU, les greenlists ne correspondent pas et ~75 % des tokens KGW ressortent faussement "clean"). Les masques ont été précalculés sur JURECA (`kgw_scores.py`, job `15399747`, ~6 min sur A100) puis rapatriés depuis `/p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre/output/kgw_{train,validation,test}.npz` via `scripts/fetch_kgw.sh` (tar streamé sur stdout d'une session SSH one-shot, plus fiable que le multiplexage `ControlMaster` qui s'est révélé instable ce jour-là — sessions acceptées côté `-O check` mais qui refusaient d'ouvrir un vrai canal).

**Branchement :** `run_smm.py --kgw output/kgw_test.npz` — les clés du `.npz` sont les `document_id` exacts du JSONL (`"validation_1"`, `"1"`, etc., pas des entiers), et le signal KGW passe par la même déduplication de n-grammes que TextSeal/Gumbel-Max (§4), avec une fenêtre de contexte de 3 tokens (le schéma KGW seed sur un 4-gramme self-salt = 3 de contexte + 1 cible, identique à TextSeal).

**Résultat :**

| | Val TPR@0.1%FPR | Score public |
|---|---|---|
| Sans KGW | 0.2822 | 0.2001 (#159) |
| Avec KGW | 0.3135 (+11 % relatif) | 0.2029 (#262, +1.4 % relatif) |

**Le gain transfère beaucoup moins bien sur test que sur val** (+11 % local vs +1.4 % public) — plus que pour la version sans KGW (où l'écart val/test était d'environ 30 % relatifs, cf. §10).

**Hypothèse `vocab_size` vérifiée et écartée (2026-07-02, log `slurm_15399747.out` rapatrié).** Le job KGW tournait en mode `--auto` (`run_kgw.sh`) : le probe compare 5 candidats sur 30 docs train labellisées, **puis le résultat est automatiquement utilisé** pour générer les masques (pas juste affiché) — ce n'était pas un oubli. Résultat du probe :

```
vocab=151643: z=9.0   vocab=151646: z=9.0   vocab=151665: z=9.3 (gagnant)
vocab=151936: z=9.2   vocab=152064: z=9.2
```

Les masques déjà utilisés (`kgw_{train,validation,test}.npz`) ont donc été générés avec **`vocab_size=151665`**, le meilleur candidat — pas `151936`. De plus, les 4 grandes tailles sont quasi équivalentes (z entre 9.0 et 9.3) : même avec le mauvais candidat, l'écart aurait été négligeable. **Cette hypothèse est donc définitivement écartée** comme explication du faible transfert val→test.

Causes restantes, non vérifiées : (a) le mix de schémas dans test diffère peut-être de val (proportions KGW/TextSeal/Gumbel-Max non garanties identiques) ; (b) sweep des `shift` KGW fait uniquement sur val (0.6/0.9/1.3), donc possible léger surajustement. Gain gardé car net et positif, mais à ne pas sur-interpréter comme pleinement validé.

---

## 9. Investigation Unigram : gelé deux fois, puis **retrouvé** (conclusion précédente renversée)

> **⚠️ Note de lecture :** les sous-sections suivantes documentent l'historique dans l'ordre chronologique, y compris une conclusion intermédiaire ("gelé définitivement") qui s'est révélée **fausse**. Le verdict final est dans la dernière sous-section : **Unigram est présent, confirmé à `vocab_size=152064`, et intégré au modèle.** L'historique est conservé parce que la méthodologie qui a conduit à l'erreur (et celle qui l'a corrigée) est instructive.

### Ce qu'on a (ce n'est pas un problème d'accès à l'info)

Le repo vendor complet est cloné et épinglé au bon commit (`vendor/unigram-watermark`, `b96cdb4`), et `watermark_config.yaml` donne tous les paramètres : `watermark_key=1873092841`, `fraction=0.5`, `strength=1.0`. Le seul paramètre absent du YAML est `vocab_size` — mais ce n'est **pas** la cause du problème (voir plus bas).

### Méthode : tester avec le vrai code vendor, pas une réimplémentation

Pour éliminer tout doute d'implémentation, le test a été refait en import direct de `gptwm.GPTWatermarkDetector` (la classe officielle du repo), pas notre `unigram_mask()` reimplémentée dans `detectors.py`.

**Découverte 1 — `vocab_size=151643` est invalide.** Le vrai tokenizer Qwen2.5-7B-Instruct a été téléchargé (révision exacte du YAML) pour vérifier : le plus grand `token_id` observé dans train+val est **151658**, alors que `151643` (taille "de base" du tokenizer, sans tokens spéciaux) est trop petite. Avec le vrai code vendor, cette valeur provoque un `IndexError` — preuve directe que c'était faux depuis le début. Nos tests précédents "marchaient" seulement parce qu'un `np.clip` silencieux masquait le dépassement d'indice, injectant du bruit sans qu'on s'en aperçoive. Tailles valides : **151665** (tokenizer complet + tokens spéciaux), **151936** (taille d'embedding du modèle Qwen2.5-7B), **152064** (candidat arrondi).

**Découverte 2 — le "signal" observé était un artefact de tokens répétés, même bug que celui déjà corrigé pour TextSeal/Gumbel-Max/KGW (§4).** Avec la méthode `.detect()` du vendor (compte les tokens verts **avec répétitions**), un z-score positif apparaissait sur les spans "non expliquées" — mais **encore plus fort côté clean** (56 faux positifs sur 355 spans clean contre 16 vrais positifs sur 208 spans wm, à `vocab=151936`, train). Un token courant répété plusieurs fois a une étiquette verte/rouge **fixe** ; le compter à chaque occurrence n'ajoute aucune évidence indépendante, ça gonfle artificiellement le score des deux côtés.

**Découverte 3 — avec la déduplication correcte, le signal disparaît entièrement.** Le vendor fournit justement `.unidetect()` (dédup sur tokens **uniques**) pour cette raison. Résultat, sur les 3 tailles de vocab valides, train et val séparément :

```
train vocab=151936  wm z_mean=0.32  clean z_mean=0.34   (quasi identiques)
val   vocab=151936  wm z_mean=0.39  clean z_mean=0.43   (quasi identiques)
```

**Conclusion : une fois mesuré correctement, il n'y a aucun signal Unigram détectable dans le pool testé.** Ce n'est plus un bug de notre côté — soit le signal n'est pas là, soit il est dilué (voir ci-dessous).

### Correction importante : ce que contient vraiment le pool "non expliqué"

Les labels train/val sont **ground truth**, pas une supposition : une span "non expliquée par TextSeal/Gumbel-Max" avec label=1 est **confirmée watermarkée**, on sait juste pas par quel schéma. Trois possibilités seulement :

- (a) vraiment **Unigram**
- (b) vraiment **KGW** (impossible à vérifier localement — nécessite CUDA Philox)
- (c) vraiment **TextSeal ou Gumbel-Max**, mais avec une réalisation anormalement faible qui a raté notre seuil `z ≥ 3` (faux négatif de notre propre heuristique de tri)

### Pourquoi il faut intégrer KGW avant de pouvoir conclure sur Unigram

Le pool résiduel est un **mélange inconnu** de (a), (b) et (c) — sans 3ᵉ détecteur fiable, impossible de les distinguer. Si le pool est dominé par du KGW non identifié (hypothèse plausible : KGW explique ~40 % des spans train), un vrai signal Unigram minoritaire serait **noyé dans la moyenne** par la masse de spans KGW à z≈0 pour Unigram. Retirer KGW du pool commun n'est possible que si on peut le calculer correctement (GPU, déjà fait sur JURECA — bloqué sur le rapatriement des `.npz`, voir §8).

**L'inverse n'est pas vrai :** KGW ne dépend pas d'Unigram. Sa détection est autonome (PRF contextuelle + greenlist propre par n-gramme), la méthode est déjà validée, seul un obstacle opérationnel (accès SSH) bloque son intégration — pas une ambiguïté de méthode comme pour Unigram.

### Test final, après intégration KGW : conclusion définitive

KGW étant maintenant intégré (§8), le plan ci-dessus a été exécuté intégralement.

**Étape 1 — purger le pool résiduel avec un vrai signal KGW.** Sur les spans watermarkées ground-truth (label=1, L≥20), classées par le premier détecteur qui les explique (z≥3) :

```
              textseal   gumbelmax   kgw   triple-résiduel   total
train (90)        83         105      55         153          396
val (90)           99          97      54         146          396
```

KGW explique **~14 % des spans** (55/396, 54/396) — bien moins que l'hypothèse de ~40 % avancée avant intégration (cette estimation datait d'une observation train seule sur TextSeal/Gumbel-Max, pas d'une mesure directe de la part KGW). Le pool résiduel se réduit tout de même nettement : ~53 %→39 % (train), ~50 %→37 % (val).

**Étape 2 — retester Unigram sur ce pool nettoyé**, avec `unidetect()` (dédup token unique, vendor officiel), 3 tailles de vocab valides, Mann-Whitney U + Kolmogorov-Smirnov (plus sensibles qu'une comparaison de moyennes à un signal minoritaire), train et val séparés :

```
train vocab=151665  wm z=0.11  clean z=-0.03  Mann-Whitney p=0.041  KS p=0.035  ← "significatif" isolé
train vocab=151936  wm z=0.34  clean z=0.34   p=0.574  p=0.448
train vocab=152064  wm z=0.24  clean z=0.18   p=0.219  p=0.522
val   vocab=151665  wm z=-0.09 clean z=-0.18  p=0.150  p=0.560  ← ne réplique PAS
val   vocab=151936  wm z=0.37  clean z=0.43   p=0.724  p=0.719
val   vocab=152064  wm z=0.19  clean z=0.12   p=0.290  p=0.550
```

Le seul résultat "significatif" (`vocab=151665`, train, p≈0.04) **ne se réplique pas sur val** (p=0.15). Sur 12 tests (3 vocabs × 2 tests statistiques × 2 splits) à α=0.05, ~0.6 faux positif est attendu par pur hasard — ce résultat isolé en est très probablement un.

**Étape 3 — vérifier l'absence de cluster caché (signal minoritaire en queue de distribution)** : les 10 z-scores les plus élevés du pool résiduel décroissent de façon lisse (train vocab=151936 : `2.92, 2.60, 2.04, 1.94...`) sans rupture nette qui indiquerait un sous-groupe réellement watermarké mélangé au bruit.

**Conclusion intermédiaire (renversée ensuite) :** à ce stade, tous les tests convergeaient vers "pas de signal" et Unigram a été déclaré gelé définitivement. **C'était faux** — voir ci-dessous pourquoi ces tests manquaient de puissance, et comment le signal a finalement été trouvé.

### Renversement : Unigram retrouvé par scan non supervisé (`unigram_scan.py`)

Deux idées nouvelles ont débloqué le problème :

**Idée 1 — la génération contraint mathématiquement le `vocab_size`.** Le `GPTWatermarkLogitsWarper` du vendor fait `new_logits = scores + strength * green_list_mask` : le masque est **additionné aux logits du modèle**, donc sa taille doit être exactement la dimension des logits. Le `config.json` du modèle épinglé (Qwen2.5-7B-Instruct, révision du YAML) donne `vocab_size = 152064`. **Une seule valeur possible** — fini le scan à l'aveugle de 4 candidats. (KGW n'a pas cette contrainte : son processor manipule des listes d'indices, pas un masque additionné, d'où un `vocab_size` indépendant.)

**Idée 2 — la greenlist étant context-free, sa présence est testable sans labels, y compris sur le test set (15× plus de données).** Statistique : max de z-score de fraction verte (tokens uniques) par fenêtre glissante aux longueurs canoniques (31–320), par document. Contrôle négatif exact : **8 puis 50 clés leurres** aléatoires sur les mêmes documents (mêmes fréquences de tokens, mêmes répétitions — null empirique parfait).

**Résultats (test set, 1 320 docs, vocab=152064) :**

```
                        vraie clé    null (50 leurres)         verdict
docs avec fenêtre z>4       28        moyenne 5.7, max 17      rang 1/51 (p≈0.02)
masse top-20 z            86.2        moyenne 78.4, max 83.3   rang 1/51 (p≈0.02)
rang moyen apparié (9 clés) 4.35      H0 : 5.00                z = 9.2
top-20 docs par z : vraie clé = rang 1 dans 100 % des cas (H0 : 11 %)
```

**Validation croisée sur train/val (labels ground truth) :** les fenêtres à z ≥ 3.5 avec la vraie clé tombent majoritairement sur des spans watermarkées (12/18), dont **7 spans résiduelles** — précisément celles qu'aucun autre détecteur n'expliquait. Exemple : `validation_50 [190,349)` z=4.20, 70 % de tokens label=1, meilleur autre détecteur z=0.85.

**Pourquoi les tests précédents rataient le signal :** (1) mauvais candidats `vocab_size` non contraints par la génération ; (2) tests de moyenne sur ~150 spans résiduelles dont seulement ~10-20 sont réellement Unigram — dilution massive (le signal individuel z≈2-4 disparaît dans la moyenne d'un pool à 90 % non-Unigram) ; (3) `strength=1.0` est un watermark **faible** (décalage par token ≈ 0.2-0.5 écart-type, contre 0.6-0.8 pour TextSeal/Gumbel), donc indétectable span par span sous z=3 la plupart du temps.

### Intégration au SMM et impact mesuré — honnêteté sur le gain

Unigram est intégré comme 4ᵉ signal (`smm_scorer.py::_unigram_signal` : masque vendor à 152064, dédup par **token unique** — un token répété ré-émet le même bit de couleur). Sweep du shift H1 sur train+val :

| Config unigram | Train TPR | Val TPR |
|---|---|---|
| sans | 0.2518 | 0.3140 |
| `(0.5,)` retenu | 0.2518 | 0.3137 |
| `(0.2, 0.35, 0.55)` | 0.2541 | 0.3110 |

**Le gain sur val est nul** (différences dans le bruit) — cohérent avec ~10-20 spans Unigram sur 396 dans val, trop faibles et trop rares pour bouger un TPR global. L'intérêt réel est sur **test**, où le scan non supervisé montre ~22 documents avec spans Unigram détectables au-dessus du null. Le signal est gardé dans le modèle car neutre sur val et positif en espérance sur test, mais il faut être lucide : **l'impact leaderboard attendu est faible** (de l'ordre de +0-1 % relatif), sans commune mesure avec le levier semi-Markov.

---

## 10. Le score val transfère-t-il tel quel au leaderboard ?

**Non — et les deux soumissions le confirment.** Raisons attendues de l'écart :

- **Risque d'optimisme du val** : pendant le débogage, j'ai inspecté les faux positifs sur validation pour diagnostiquer le bug de n-grammes répétés (§4), et les hypothèses de `shift` KGW ont été sweepées sur val (§8). Corriger un vrai bug de signal est légitime, mais le pipeline final a de fait été affiné en observant le comportement sur val — un optimisme est donc attendu, pas une garantie de transfert 1:1.
- **Taille et bruit du budget FPR** : val n'a que 90 documents → budget de 51 tokens clean pour 0.1 % FPR, un échantillon petit et bruyant. Test a 1 320 documents (~15×), donc un budget bien plus large et une estimation plus stable statistiquement, mais qui peut différer selon le mix réel de schémas/longueurs dans test.
- **Écarts observés** : sans KGW, 0.2822 (val) → 0.2001 (test public), baisse d'environ 30 % relatifs. Avec KGW, le gain val (+11 %) ne transfère que très partiellement au public (+1.4 %) — voir §8 pour les hypothèses. L'ordre de grandeur global (×28 vs la précédente meilleure soumission HMM) reste ce qui compte le plus ici.
- Le score officiel final utilisera par ailleurs un **sous-ensemble caché** de test révélé après la clôture des soumissions (mentionné dans la réponse API) — le score public actuel (0.2029) n'est donc pas nécessairement le score final.

---

## 11. Prochaines étapes, par impact attendu

1. ~~Comprendre l'écart val/test spécifique à KGW : vérifier le `vocab_size`~~ — **fait, écarté** (§8) : le probe a tourné en mode `--auto` et a correctement sélectionné `151665` pour générer les masques déjà utilisés ; les 4 grandes tailles candidates sont de toute façon quasi équivalentes (z entre 9.0 et 9.3). Ce n'est pas la cause du faible transfert val→test. Cause réelle encore non identifiée.
2. **Cross-validation train+val (180 docs)** pour réduire le bruit de l'estimation TPR@0.1%FPR utilisée pour piloter les choix de modèle — priorité désormais la plus haute (fondation nécessaire avant tout raffinement de modèle, cf. discussion).
3. **LLR non-paramétrique** (histogrammes fittés sur train+val) à la place de la grille à 3 shifts gaussiens par schéma.
4. **Réestimation supervisée des priors** (longueur de spans, `p_span`, bords tronqués) au lieu de valeurs fixées à la main.
5. **Pondération par entropie** (forward pass `Qwen2.5-7B-Instruct` sur GPU, comme dans le papier TextSeal) pour concentrer le signal sur les tokens à haute incertitude.
6. **Couverture géométrique** (`localized_detect` du vendor TextSeal) pour les spans très diluées dans un document mixte.

**Unigram : résolu et intégré** (§9) — `vocab=152064`, signal faible confirmé par scan non supervisé. Ne pas s'attendre à un gain leaderboard important (spans rares et faibles), mais le détecteur est correct.

**Leçon méthodologique retenue :** deux conclusions "définitives" successives sur Unigram étaient fausses. Ce qui a fini par marcher : (1) contraindre les paramètres par la **physique de la génération** plutôt que scanner des candidats ; (2) tester sur **toutes les données disponibles** (test set non labellisé inclus) avec un **null empirique** (clés leurres) plutôt que des tests de moyenne sur des petits pools dilués.

---

## 12. Comment exécuter

### Local (CPU) — semi-Markov, sans KGW

Prérequis : dataset dans `data/watermark_localization/`, Python avec `torch`, `numpy`, `scipy`.

```bash
cd cispa_final/task_1_text_watermark/alexandre

python run_smm.py --split validation --out val_scores.jsonl
python run_smm.py --split test --out submission.jsonl

python ../../shared/task1_eval.py \
    --dataset ../../data/watermark_localization/validation.jsonl \
  --predictions val_scores.jsonl \
  --method "semi-Markov segment posterior (TextSeal+GumbelMax), ngram-dedup"

python ../../shared/submit.py submission.jsonl \
    --task-id 30-watermark-localization --action submit --owner ansart1 \
  --method "semi-Markov segment posterior, no KGW"
```

### Avec KGW

```bash
# 1. précalcul GPU sur JURECA (déjà fait, job 15399747)
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre
sbatch run_kgw.sh   # -> output/kgw_{train,validation,test}.npz

# 2. rapatriement en local (one-shot SSH + TOTP, plus fiable que ControlMaster)
$env:TOTP_CODE="<code>"; bash cispa_final/scripts/fetch_kgw.sh
# -> task_1_text_watermark/alexandre/kgw_bundle.tar (à extraire dans output/)

# 3. scoring avec KGW
python run_smm.py --split test --out submission.jsonl --kgw output/kgw_test.npz
```

---

## 13. Carte des fichiers

| Fichier                              | Rôle                                                          |
| --------------------------------------- | ---------------------------------------------------------------- |
| `detectors.py`                        | Signaux PRF CPU (TextSeal, Gumbel-Max) + dédup n-grammes |
| `unigram_scan.py`                     | Greenlist Unigram (vocab 152064) + scan non supervisé par fenêtres/clés leurres |
| `smm_scorer.py`                       | **Modèle retenu** : semi-Markov à segments, forward-backward (TS+GM+KGW+Unigram) |
| `run_smm.py`                          | CLI : score un split → JSONL, précision pleine, `--kgw` optionnel |
| `kgw_scores.py`                       | Masques green KGW (CUDA Philox obligatoire, calculés sur JURECA) |
| `run_kgw.sh`                          | SLURM : précalcul KGW                                            |
| `../../scripts/fetch_kgw.sh`          | Rapatriement one-shot des `.npz` KGW depuis JURECA (tar sur stdout) |
| `matched_filter.py`                   | *(abandonné)* fenêtres discrètes + `min(g,d)`, remplacé par v2  |
| `hmm_scorer.py`, `run_hmm.py`         | *(abandonné v1)* HMM token-à-token, gardé en référence          |
| `build_scores.py`                     | *(abandonné)* fusion z-score                                     |
| `features.py`, `train_calibrator.py`  | *(abandonné)* régression logistique                               |

**Sorties :** `alexandre/submission_smm_kgw.jsonl` (soumis, `submission_id=262`, meilleur score), `submission_smm_nokgw.jsonl` (`submission_id=159`), `val_scores_kgw.jsonl`, `output/kgw_*.npz`.
