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
| **Soumission #159 (semi-Markov, sans KGW)**    | **score public 0.2001** — nouveau best, ×28 vs #101                      |
| Val locale TPR@0.1%FPR — HMM v1                | 0.0185 (float) / 0.0140 (arrondi 6 déc.)                                  |
| Val locale TPR@0.1%FPR — semi-Markov v2        | **0.2822**                                                                |
| KGW dans v2                                    | ❌ pas encore intégré — fichiers restés sur JURECA, session SSH expirée   |
| Unigram                                        | ❌ gelé — greenlist non reproduite malgré plusieurs variantes testées     |

**Meilleur score leaderboard public : 0.2001** (soumission #159, `submission_id=159`). Le gain principal vient d'un changement de modèle (HMM → semi-Markov à longueurs discrètes) et de la correction d'un bug de signal (n-grammes répétés), pas d'un détecteur supplémentaire.

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
| **Unigram**    | 1 si token ∈ greenlist fixe                                    | `watermark_key`, `fraction=0.5`, `vocab_size=151643`                | Bernoulli(0.5)         |
| **KGW**        | *pas calculé en local* — nécessite CUDA Philox                | `gamma=0.25`, schéma `ff-anchored_minhash_prf-4-True-1306382177`   | Bernoulli(0.25)        |

**Implémentation PRF :** chargement direct de `vendor/textseal/textseal/watermarking/core.py` (`prf_uniform`) via `importlib`, sans importer le package `textseal` complet (évite la dépendance `nltk`).

**Unigram — gelé, avec preuve.** J'ai retesté sérieusement cette piste : 2 méthodes de génération de greenlist (`np.random.default_rng(key)` direct, et seedé par `sha256(key)`) croisées avec 4 tailles de vocabulaire (151643 / 151665 / 151936 / 152064), évaluées sur les spans watermarkées *non expliquées* par TextSeal/Gumbel-Max — **séparément sur train et sur val**. Résultat : aucune variante ne montre de séparation stable entre spans watermarkées et clean sur les deux splits à la fois (ex. `sha256`@151936 : z≈1.2 sur wm mais z≈1.4 sur clean — corrélation de bruit, pas un greenlist). Une piste antérieure (session précédente) rapportait un z=5.3 pour une variante proche ; elle ne s'est pas reproduite sous ce protocole train/val séparé, probablement mesurée sans tenue hors-échantillon. **Unigram reste exclu du modèle** faute de signal confirmé.

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
- **Émission d'une span** : modèle de décalage gaussien sur les signaux standardisés (z-scores sous H0). Pour un décalage hypothétique `shift` :

  ```
  LLR(s, e, schéma) = shift · Σx[s:e] − (e−s) · shift² / 2
  ```

  combiné par `logsumexp` sur 3 hypothèses de `shift` par schéma (valeurs estimées sur train : TextSeal ≈ 0.45–0.9, Gumbel-Max ≈ 0.55–1.1) et sur les schémas disponibles (TextSeal, Gumbel-Max ; KGW prêt mais pas encore branché — voir §8).
- **Spans tronquées** : gérées explicitement par des hypothèses préfixe `[0, e)` et suffixe `[s, n)` avec un prior dédié, pour les documents qui commencent ou finissent en plein milieu d'un watermark.
- **Score par token** : posterior exact `P(token appartient à un segment watermarké | document)`, calculé par forward-backward en log-espace (`scipy.special.logsumexp`), converti en log-odds puis compressé par une sigmoïde à température (température = 40, choisie pour garder une résolution fine sans jamais saturer à 0/1 exactement).

### Pourquoi ça résout le boundary bleed sans heuristique manuelle

Un token profond dans une span appartient à très peu de segmentations alternatives plausibles (toutes le couvrent). Un token de bordure appartient à des segmentations où il est "dans" la span *et* à des segmentations où il n'y est pas (span plus courte) — le posterior fait cette moyenne automatiquement, pondérée par le prior de longueur. C'est ce que l'heuristique `min(gauche, droite)` visait à approximer manuellement (voir §7) ; le calcul exact fait mieux.

### Résultat

| Version                          | Val TPR@0.1%FPR | Score public |
| ---------------------------------- | ------------------ | -------------- |
| HMM v1 (sans KGW, arrondi 6 déc.) | 0.0140              | 0.0065 (#30)   |
| HMM v1 + KGW                      | 0.0066 ⚠️          | 0.0071 (#101)  |
| **Semi-Markov v2 (sans KGW, précision pleine)** | **0.2822** | **0.2001 (#159)** |

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

## 8. Pourquoi la soumission actuelle est sans KGW

KGW nécessite `torch.randperm` sur un `torch.Generator` **CUDA** (Philox) pour reproduire exactement la génération des organisateurs — impossible à recalculer sur CPU (sur CPU, les greenlists ne correspondent pas et ~75 % des tokens KGW ressortent faussement "clean"). Les masques ont été précalculés sur JURECA lors d'une session précédente (`kgw_scores.py`, job `15399747`, ~6 min sur A100), mais :

1. Le fichier de sortie `kgw_{train,validation,test}.npz` et le log du probe (`slurm_15399747.out`, qui contient le `vocab_size` retenu) sont restés sur le cluster.
2. La connexion SSH multiplexée (`ControlMaster`) utilisée pour éviter de redemander un TOTP à chaque appel a expiré entre les deux sessions de travail.
3. Rapatrier ces fichiers nécessite donc un nouveau TOTP — **prochaine étape**, en attente.

Le code de `smm_scorer.py` est déjà prêt à recevoir KGW comme signal supplémentaire (`extra={"kgw": mask_array}`, avec déduplication de n-grammes déjà branchée pour ce schéma aussi). Environ 40 % des spans train ne sont expliquées ni par TextSeal ni par Gumbel-Max (probablement du KGW pur) — c'est le principal levier restant.

---

## 9. Le score val (0.2822) transfère-t-il tel quel au leaderboard ?

**Non — et le résultat obtenu (0.2001 public) le confirme.** Raisons attendues de l'écart :

- **Risque d'optimisme du val** : pendant le débogage, j'ai inspecté les faux positifs sur validation pour diagnostiquer le bug de n-grammes répétés (§4). Corriger un vrai bug de signal est légitime, mais le pipeline final a de fait été affiné en observant le comportement sur val — un léger optimisme est donc attendu, pas une garantie de transfert 1:1.
- **Taille et bruit du budget FPR** : val n'a que 90 documents → budget de 51 tokens clean pour 0.1 % FPR, un échantillon petit et bruyant. Test a 1 320 documents (~15×), donc un budget bien plus large et une estimation plus stable statistiquement, mais qui peut différer selon le mix réel de schémas/longueurs dans test.
- **Écart observé** : 0.2822 (val) vs 0.2001 (test public) — une baisse d'environ 30 % relatifs, cohérente avec un effet d'optimisme modéré plutôt qu'un problème de généralisation grave. L'ordre de grandeur (×28 vs la précédente meilleure soumission) est ce qui compte le plus ici.
- Le score officiel final utilisera par ailleurs un **sous-ensemble caché** de test révélé après la clôture des soumissions (mentionné dans la réponse API) — le score public actuel (0.2001) n'est donc pas nécessairement le score final.

---

## 10. Prochaines étapes, par impact attendu

1. **Intégrer KGW** dans le semi-Markov (fichiers déjà calculés, juste à rapatrier — TOTP nécessaire). Plus gros levier restant : ~40 % des spans train ne sont expliquées par aucun des deux schémas actuellement modélisés.
2. **Vérifier le probe KGW** (`slurm_15399747.out`) pour confirmer le `vocab_size` retenu, et tempérer les émissions KGW (facteur `shift` plus faible que TextSeal/Gumbel si le signal binaire s'avère plus faible que prévu) avant de resoumettre.
3. **Apprentissage des priors de longueur / shifts par EM (Baum-Welch semi-Markov)** au lieu de valeurs fixes estimées à la main sur train.
4. **Cross-validation train+val (180 docs)** pour réduire le bruit de l'estimation TPR@0.1%FPR utilisée pour piloter les choix de modèle.
5. **Pondération par entropie** (forward pass `Qwen2.5-7B-Instruct` sur GPU, comme dans le papier TextSeal) pour concentrer le signal sur les tokens à haute incertitude.
6. **Couverture géométrique** (`localized_detect` du vendor TextSeal) pour les spans très diluées dans un document mixte.

---

## 11. Comment exécuter

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

### Avec KGW (une fois les `.npz` rapatriés)

```bash
python run_smm.py --split test --out submission.jsonl --kgw output/kgw_test.npz
```

### JURECA — précalcul KGW (déjà fait, job `15399747`)

```bash
jutil env activate -p training2625
cd /p/home/jusers/ansart1/jureca/code/cispa_final/task_1_text_watermark/alexandre
sbatch run_kgw.sh   # -> output/kgw_{train,validation,test}.npz
```

---

## 12. Carte des fichiers

| Fichier                              | Rôle                                                          |
| --------------------------------------- | ---------------------------------------------------------------- |
| `detectors.py`                        | Signaux PRF CPU (TextSeal, Gumbel-Max, Unigram) + dédup n-grammes |
| `smm_scorer.py`                       | **Modèle retenu** : semi-Markov à segments, forward-backward     |
| `run_smm.py`                          | CLI : score un split → JSONL, précision pleine                  |
| `kgw_scores.py`                       | Masques green KGW (CUDA Philox obligatoire, calculés sur JURECA) |
| `run_kgw.sh`                          | SLURM : précalcul KGW                                            |
| `matched_filter.py`                   | *(abandonné)* fenêtres discrètes + `min(g,d)`, remplacé par v2  |
| `hmm_scorer.py`, `run_hmm.py`         | *(abandonné v1)* HMM token-à-token, gardé en référence          |
| `build_scores.py`                     | *(abandonné)* fusion z-score                                     |
| `features.py`, `train_calibrator.py`  | *(abandonné)* régression logistique                               |

**Sorties :** `alexandre/submission_smm_nokgw.jsonl` (soumis, `submission_id=159`), `val_smm_scores.jsonl`, `kgw_*.npz` (sur JURECA).
