## **TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection** 

**Tom Sander** _[⋆,][†]_ , **Hongyan Chang** _[†]_ , **Sylvestre-Alvise Rebuffi** , **Tomáš Souček** , **Tuan Tran** , **Valeriu Lacatusu** , **Alexandre Mourachko** , **Surya Parimi** _[§]_ , **Christophe Ropers** _[§]_ , **Rashel Moritz** _[§]_ , **Vanessa Stark** _[§]_ , **Hady Elsahar** _[†]_ , **Pierre Fernandez** _[⋆,][†]_ 

FAIR, Meta Superintelligence Labs 

> _⋆_ Equal contributors, _†_ Core team, _§_ Project Support. 

We introduce _TextSeal_ , a state-of-the-art watermark for large language models. Building on Gumbel-max sampling, TextSeal introduces dual-key generation to restore output diversity, along with entropy-weighted scoring and multi-region localization for improved detection. It supports serving optimizations such as speculative decoding and multi-token prediction, and does not add any inference overhead. TextSeal strictly dominates baselines like SynthID-text in detection strength and is robust to dilution, maintaining confident localized detection even in heavily mixed human/AI documents. The scheme is theoretically distortion-free, and evaluation across reasoning benchmarks confirms that it preserves downstream performance; while a multilingual human evaluation (6,000 A/B comparisons, 5 languages) shows no perceptible quality difference. Beyond its use for provenance detection, TextSeal is also “radioactive”: its watermark signal transfers through model distillation, enabling detection of unauthorized use. 

**Correspondence:** tomsander@meta.com, pfz@meta.com **Code:** https://github.com/facebookresearch/textseal 

## **1 Introduction** 

The rapid adoption of LLMs in production systems has created a need for reliable provenance mechanisms. Watermarking, by embedding an imperceptible, algorithmically-detectable signal during the generation, addresses several needs at once: detecting AI-generated content, complying with regulations that mandate machine-detectable marking of AI outputs (Eur, 2024; European Commission, 2026), and enabling applications such as monitoring model output usage, preventing self-training on generated data, and detecting unauthorized distillation (Sander et al., 2024; Sablayrolles et al., 2020). 

For production deployment, it is highly desirable to use _distortion-free_ watermarking, which ensures that next-token selection follows exactly the same distribution as that produced by the LLM. It pre- 

**==> picture [434 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
Task None TextSeal<br>TextSeal<br>8 5 Gumbel-Max AIME 40.1 41.1<br>- __ ee - 4h ie SynthID MATH 79.8 79.8<br>6 A __ 4 e =o GSM8KHumanEval 97.095.4 93.396.0<br>3 MBPP 50.2 49.2<br>| 4 aLaant * I 2 _------ASNex | RRRRN RSE esi p ee  = 0.01 ARC-CARC-E 88.393.4 88.593.7<br>GPQA 50.5 50.0<br>2 TextSeal (high) SynthID HellaSwag 94.7 94.8<br>TextSeal No watermark 1 MMLU 49.2 51.5<br>Vv $ Gumbel-Max v ————=—=——, SQA 15.8 16.0<br>0 0 WinoGrande 93.2 93.5<br>0.2 0.3 0.5 1 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0<br>Diversity (Self-BLEU)  a Document length (k tokens) Average 70.6 70.6<br>(a) Diversity–detectability trade-off. (b) Detectability under dilution. (c) Performance.<br>)  )<br> p  p<br>10 10<br>log log<br>Detectability ( Detectability (<br>**----- End of picture text -----**<br>


**Figure 1 TextSeal achieves state-of-the-art detectability while preserving generation diversity and downstream performance** (Qwen3.5-27B). **(a)** TextSeal strictly dominates SynthID across the diversity-detectability frontier (ELI5, 400 tokens, _T_ =0 _._ 8, top- _p_ =0 _._ 9). **(b)** Localized detection remains confident even at 10 _×_ dilution, where global baselines fail. **(c)** Accuracy across 12 benchmarks is preserved ( _T_ =0 _._ 6). 

1 

serves the exact decoding configuration (temperature, top- _p_ ) the model was tuned for, embedding the watermark at zero cost to any individual generation’s quality. A recent large-scale comparison (Fernandez et al., 2025) shows that the Gumbel-max watermark (Aaronson and Kirchner, 2023) achieves the best detectability-quality Pareto frontier by far among other methods, e.g., green-red list (Kirchenbauer et al., 2023a), SynthID (Dathathri et al., 2024), DiPMark (Wu et al., 2023). However, Gumbel-max has one important drawback: it is fully _deterministic_ (a fixed prompt and secret key always produce the same output, eliminating diversity), which can in turn trigger degenerate loops when repeated n- grams used for hashing cause the pseudo-random function to lock onto the same token (Remark 1). For instance, SynthID (deployed in Google’s Gemini) resolves the determinism while remaining distortionfree, with a tournament-sampling design. 

_TextSeal_ is a distortion-free, non-deterministic watermark for LLMs. It builds upon the Gumbel-max framework and introduces three core improvements: 

1. **Dual-KeyGeneration:** We overcome determinism by randomly alternating between two secret keys during generation, restoring diversity at low cost to detection power. This natively supports speculative decoding and Multi-Token Prediction (MTP) without additional latency (subsection 3.1). 

2. **Entropy-Weighted Detection:** We introduce tests tailored to the dual-key generation, that may leverage the entropy of a proxy model, and moment-matched Gamma approximations to have calibrated _p_ -values (subsection 3.2). 

3. **LocalizedDetection:** We identify individual watermarked segments within a document via a multiregion geometric cover search, dramatically boosting detection under dilution (subsection 3.3). 

As summarized in Figure 1, TextSeal achieves state-of-the-art detectability while offering a superior diversity-detectability trade-off compared to existing methods. Our localized detection is robust to dilution within long documents, and TextSeal preserves the downstream capabilities of the model across 12 complex benchmarks. TextSeal adds only _≤_ 0 _._ 3% sampling overhead (3 _×_ faster than SynthID; subsection 5.4). Beyond provenance, TextSeal is _radioactive_ (Sander et al., 2024; Sablayrolles et al., 2020): the watermark signal transfers through model distillation, meaning that a student model trained on watermarked outputs inherits a detectable trace. This provides a practical safeguard against unauthorized distillation and enables monitoring of how model outputs are used downstream (in training pipelines, RAG systems, or by competitors). We demonstrate this experimentally in section 6. 

The paper is organized as follows. Section 2 presents the technical background . Section 3 describes the TextSeal method. Section 4 presents the main experimental results. Section 5 provides ablation studies and additional analyses. Section 6 demonstrates watermark transfer through distillation. 

## **2 Background and Related Work** 

## **2.1 LLM Watermarking** 

Early text watermarking relied on edit-based methods (Topkara et al., 2005, 2006c) with low robustness. For LLMs, two concurrent approaches appeared after ChatGPT: green-red list (Kirchenbauer et al., 2023a) and Gumbel-max sampling (Aaronson and Kirchner, 2023), both using pseudorandom seeds from a secret key and preceding tokens, enabling lightweight detection without access to the model. Some subsequent work explores multi-bit watermarking (Fernandez et al., 2023; Yoo et al., 2024; Qu et al., 2024), undetectable constructions (Christ et al., 2023; Kuditipudi et al., 2023), low-entropy optimizations (Lee et al., 2023; Huang et al., 2023), adaptive green-red variants (Wang et al., 2025), distillation for open-weights model (Gu et al., 2023), etc. See subsection A.3 for detailed scheme descriptions. Semantic watermarks (Liu et al., 2023; Liu and Bu, 2024; Hou et al., 2023) offer increased robustness to adversaries, but require auxiliary semantic encoders. This makes them harder to deploy, which is the reason why we do not consider them in the remaining of the work. 

Beyond detection, watermark radioactivity (Sander et al., 2024) has been leveraged for data protection (RAG (Jovanović et al., 2025), contamination (Sander et al., 2025), copyright (Zhang et al., 2025)), which we extend in section 6 to reasoning-trace distillation. 

2 

## **2.2 Distortion-Freeness and Choice of Baselines** 

In the literature on LLM watermarking, schemes are typically divided into two families: _distortionary_ (biased) and _distortion-free_ (unbiased/distribution-preserving). The key distinction is whether the watermark alters text quality. A watermarking sampler is _distortion-free_ (or _non-distortionary_ ) if, in expectation over the random seed, it outputs each token with exactly its original LLM probability: no token is favored or suppressed by the watermark. This corresponds to the _single-token non-distortion_ property of Dathathri et al. (2024); formal definitions and the stronger _single-sequence_ variant are given in subsection A.5. 

For instance, green-red list (Kirchenbauer et al., 2023a) and low-entropy filtering methods, like SWEET (Lee et al., 2023) which skips watermarking on low-entropy tokens, are _distortionary_ : they shift the output distribution, degrading generation. MorphMark (Wang et al., 2025) adaptively scales the greenred bias based on the natural green-list probability mass, reducing distortion in low-entropy contexts, but remains distortionary since it still applies a logit bias. Gumbel-max (Aaronson and Kirchner, 2023), Permute-and-Flip (Zhao et al., 2024), DiPMark (Wu et al., 2023) (non-distortionary green-red via pseudorandom permutations), SynthID-Text (Dathathri et al., 2024) (deployed in Google Gemini and detailed in subsection A.2), and WaterMax (Giboulot and Furon, 2024) (multiple generations per query, impractical for production) are non-distortionary methods. 

Aligned with recent large-scale evaluations (Fernandez et al., 2025), we found that Gumbel-max and SynthID achieved the best detectability-quality Pareto frontier. Therefore, we build TextSeal upon Gumbel-Max (which we detail next), and compare against these two baselines. Because all three are single-token non-distortionary, we can fix the LLM, temperature, and top- _p_ , and vary only their watermark-specific diversity parameter, isolating the watermark’s effect. 

## **2.3 Gumbel-Max Watermarking** 

We consider a language model generating a sequence of tokens. At each time step _t_ , the model predicts a probability distribution _**p**_[(] _[t]_[)] = ( _p_ 1 _, . . . , p|V|_ ) over the vocabulary _V_ . Let _K_ be a secret key used for watermarking and _ht_ the context (history of tokens) at step _t_ . The goal of watermarking is to select a token _xt_ such that its selection is statistically correlated with a pseudo-random value derived from _ht_ and _K_ , while preserving the original distribution _**p**_ (single-token non-distortion, Definition 1). This concept was introduced for LLMs by Aaronson and Kirchner (2023) with the Gumbel-max scheme. 

## **2.3.1 Gumbel-max mechanism** 

The standard Gumbel watermarking ensures detectability by making the sampling process deterministic given the secret key and watermark context (see App. Fig 10 for an overview). 

_Embedding._ At each generation step _t_ , the watermark operates on a _watermark context window_ **w** = ( _xt−k, . . . , xt−_ 1), consisting of the _k_ last generated tokens. This window, together with the secret key _K_ , seeds a Pseudo-Random Function (PRF) that assigns a pseudo-random value _Rv ∈_ [0 _,_ 1] to every candidate token _v_ in the vocabulary: 

**==> picture [90 x 11] intentionally omitted <==**

The PRF is deterministic: for a given context window, secret key, and candidate token, it always returns the same value. However, its output is indistinguishable from uniform randomness to anyone who does not know _K_ (see subsection A.1 for implementation details of the PRF). 

The watermark then selects the next token by combining these pseudo-random values with the LLM’s probability distribution. Concretely, it picks: 

**==> picture [94 x 20] intentionally omitted <==**

where _pv_[(] _[t]_[)] is the probability assigned to token _v_ by the LLM at step _t_ . This balances two factors: tokens with high model probability _pv_ are naturally favored, but among tokens of similar probability, 

3 

the one with the highest PRF value _Rv_ wins. This creates a statistical correlation between the chosen tokens and the secret key, which can later be detected. 

This selection rule is equivalent to two well-known sampling schemes: 

- **Inverse Transform Method:** Sort tokens by descending probability, compute the CDF, and select the token corresponding to the quantile _u_ = PRF( **w** _, K_ ). 

- **Gumbel-Max Trick:** Sample Gumbel noise _Gv_ = _−_ log( _−_ log( _Rv_ )) for each token and select _xt_ = arg max _v_ ( _Gv_ + log _pv_[(] _[t]_[)][)][.] 

Put differently, the watermarking scheme samples from the original distribution _**p**_ , but uses a deterministic source of randomness derived from the secret key and context, instead of true randomness. This is what gives it the single-token non-distortion property (Definition 1), as formalized in Proposition 1 below. 

_Detection._ Given a text, the detector re-computes the PRF values using the secret key and the preceding tokens, then checks whether the score is higher than expected by chance. 

We denote by _x_[(1)] _, . . . , x_[(] _[T]_[ )] the sequence of tokens in the text, and by _**R**_[(] _[t]_[)] _∈_ [0 _,_ 1] _[|V|]_ the key random vector re-computed from the _k_ preceding tokens and the secret key. We define _Rt_ := _Rx_[(] _[t]_[(][)] _[t]_[)][,][the][PRF] value of the token selected at time-step _t_ . The detection score is calculated as: 

**==> picture [100 x 30] intentionally omitted <==**

Intuitively, watermarked tokens tend to have high _Rt_ values (since the selection rule favors them), making _−_ ln(1 _− Rt_ ) large. For unwatermarked text, _Rt_ values are essentially random, yielding a lower score. A statistical test then determines whether the observed score is significantly higher than expected under the null hypothesis _H_ 0 (no watermark). In practice, we choose a threshold _τ_ (depending on the desired false positive rate) and flag a text as watermarked if _ST > τ_ . 

## **2.3.2 Theoretical Properties** 

The following results formalize the two key guarantees of Gumbel-max watermarking: single-token nondistortion and detectability. The proofs are not original contributions; they were presented by Aaronson and Kirchner (2023) and formalized by Fernandez et al. (2023). We provide them in Appendix B. 

## **2.3.3 Single-Token Non-Distortion** 

**Proposition 1** (Sampling probability) **.** _Consider a discrete distribution_ _**p**_ = ( _p_ 1 _, . . . , pV_ ) _and V_ = _|V| iid random variables_ _**R**_ = ( _R_ 1 _, . . . , RV_ ) _s.t. Rv ∼U_ [0 _,_ 1] _. Let V[⋆]_ = arg max _v Rv_[1] _[/p][v] . Then:_ 

**==> picture [72 x 11] intentionally omitted <==**

**Corollary 1.** _Conditionally on V[⋆]_ = _v, RV ⋆ ∼ Beta_ (1 _/pv,_ 1) _._ 

Proposition 1 establishes that Gumbel-max is single-token non-distortionary (Definition 1): in expectation over the random key, the selected token follows exactly the LLM’s original distribution. The corollary characterizes the distribution of the PRF value for the selected token, which is useful for the detection analysis below. 

**Remark 1** (Repeated _n_ -grams and single-sequence non-distortion) **.** Single-token non-distortion (Proposition 1) does not guarantee that a full _sequence_ is distributed as the original LLM. Dathathri et al. (2024) define a stronger property: _single-sequence non-distortion_ (Definition 2), requiring that the joint probability of a complete response is preserved: E _k_ [ _P_ wm( _**y** |_ _**x** , k_ )] = _p_ LM( _**y** |_ _**x**_ ). This is violated whenever the same _k_ -gram context repeats within a generation: the PRF produces identical values, so the same token is always re-selected, creating unwanted correlations in the sequence distribution. 

To restore single-sequence non-distortion, we can apply _repeated context masking_ (Dathathri et al., 2024): a set _S_ of seen context windows is maintained per generation; on the first occurrence the 

4 

watermark is applied, on subsequent occurrences the sampler falls back to standard unwatermarked sampling. Our main evaluations do not enforce this protocol (except in subsection E.1), as repeated _k_ -grams are rare in practice with _k ≥_ 3. 

## **2.3.4 Detectability** 

**Proposition 2** ( _p_ -value under _H_ 0) **.** _Under H_ 0 _(text not watermarked), the score ST follows a_ Γ( _T,_ 1) _distribution. The p-value associated to a score s is:_ 

**==> picture [308 x 25] intentionally omitted <==**

_where_ Γ( _T, s_ ) _is the upper incomplete gamma function._ 

This provides an _exact_ false positive rate: given any desired significance level _α_ , we can compute a detection threshold _τ_ such that the probability of wrongly flagging unwatermarked text as watermarked is exactly _α_ . 

**==> picture [452 x 25] intentionally omitted <==**

_where HT_ = _−_[�] _[T] t_ =1 _[p][t]_[ ln(] _[p][t]_[)] _[is][the][entropy][of][the][completion.]_ 

_This bound reveals that detectability scales with the_ entropy _of the generated text._ When the LLM is uncertain (high entropy), many tokens have non-negligible probability, giving the watermark more room to influence the selection and producing a stronger signal. Conversely, when the model is very confident (low entropy), the top token dominates regardless of the PRF values, and the watermark signal is weak. This entropy dependence motivates the entropy-weighted detection of subsection 3.2. 

**==> picture [434 x 258] intentionally omitted <==**

**----- Start of picture text -----**<br>
Embedding Detection<br>w LLM context<br>x 1 x 2 · · · xt - k · · · xt -1 x 1 x 2 · · · xi - k · · · xi -1 xi · · · xn<br>Candidates v ∈V<br>LLM context w i<br>v 1 v 2 .. vV<br>e.g., small,<br>Route Each v quantized<br>1 −α α PRF( k [(1)] ) PRF( k [(2)] ) LLMproxy<br>PRF( k [(1)] ) PRF( k [(2)] ) s [(1)] i =  − ln(1 −Ri [(1)] ) s [(2)] i<br>Hi → wi [ent]<br>LLM si = (1 −α )  s [(1)] i +  α s [(2)] i<br>R [(] [j] [)]<br>p [(] [t] [)]<br>Reweight: s ˜ i =  wi [ent] · si<br>xt = arg max v Rv [(] [j] [)] [,] [1] [/p] [(] v [t] [)] si ˜<br>xt P( xt = i ) = pi candidateDyadic p raw p raw L 2 L 00<br>(distortion-free) windows 4 L 0<br>x 1 xn<br>Region 1 Region 2<br>p < 10 [−] [6] p < 10 [−] [4]<br>**----- End of picture text -----**<br>


**Figure 2** TextSeal overview. **Left (Embedding):** At each step, one of two keys is randomly selected (probability _α_ for _k_[(2)] , 1 _−α_ for _k_[(1)] ), and the token is chosen via Gumbel-Max using the selected key’s PRF (subsection 3.1). **Right (Detection):** Scores are computed under both keys and fused per-token, weighted by entropy (subsection 3.2), then a geometric cover search localizes watermarked regions (subsection 3.3). 

5 

## **3 Method: TextSeal** 

TextSeal addresses three key limitations of the standard Gumbel-max watermark: its deterministic outputs, its suboptimal detection in mixed-entropy text, and the lack of localized detection capability. We describe each improvement below, and present an overview in Figure 2. 

## **3.1 Dual-Key Routing for Diversity and Speculative Decoding** 

Gumbel-Max is deterministic: for a given context and secret key, the output token is fixed, so regenerating the same prompt always produces identical text, limiting user experience and triggering repetitive loops (Holtzman et al., 2019). TextSeal addresses this by maintaining two secret keys _k_[(1)] and _k_[(2)] that restore diversity while preserving both detectability and single-token non-distortion (Definition 1). 

_Embedding._ At each generation step _t_ , one key is selected at random: _k_[(1)] with probability 1 _− α_ , or _k_[(2)] with probability _α_ . The token is produced via Gumbel-Max using the selected key’s PRF: 

**==> picture [316 x 19] intentionally omitted <==**

The routing probability _α ∈_ [0 _,_ 0 _._ 5] controls the diversity-detectability trade-off: _α_ = 0 is the original single-key scheme with maximum detectability but no diversity, while _α_ = 0 _._ 5 routes evenly between the two keys. Dual-key routing also doubles the tolerance to repeated _n_ -grams before single-sequence non-distortion is compromised (Remark 1): when a context window appears for the first time it is watermarked with one key; if it recurs, the other key is used; only on a third occurrence must the sampler fall back to unwatermarked sampling. 

_Detection._ The detector does not know which key generated each token. To capture signal from both potential paths, we compute scores under both keys and aggregate them as a weighted sum: 

**==> picture [348 x 14] intentionally omitted <==**

We call this strategy “early fusion”, in contrast with methods that would compute two p-values and aggregate them later. Under _H_ 0, _si_ is a weighted combination of independent exponentials with mean 1 and variance _θR_ = _α_[2] + (1 _− α_ )[2] . The final p-value is computed using the unified framework in subsection 3.2. We show that this early-fusion approach is better than Fisher or Bonferroni aggregations in subsubsection C.1.1, and support it empirically in subsection 5.1. 

_Compatibility with speculative decoding._ In speculative decoding (Leviathan et al., 2023), a draft model _PD_ proposes tokens accepted by a target model _PT_ . With dual-key watermarking, the draft uses _k_[(1)] and rejected correction tokens use _k_[(2)] . The draft acceptance rate naturally determines the routing ratio _α_ . Since this rate varies by domain and model pair, _α_ can be calibrated at detection time or set to 0 _._ 5 as a robust default that makes detection method invariant to the true routing ratio. This extends naturally to Multi-Token Prediction (MTP) (Gloeckle et al., 2024), where all _K_ auxiliary heads share _k_[(1)] and fall back to _k_[(2)] , preventing the fracturing of statistical power across many keys. 

## **3.2 Entropy-Weighted Detection** 

_Entropy Weighting._ When the next-token distribution has low entropy, the top token already has probability close to 1, so the choice is weakly influenced by the PRF value _Rv_ and carries little watermark signal. TextSeal therefore weights each token’s detection score by its local entropy _Hi_ , so that high-entropy positions contribute more to the final statistic. We estimate _Hi_ with a single forward pass of an auxiliary model, e.g., a smaller or quantized model from the same family as the generator. 

Formally, we assign each token-level score _si_ an entropy weight _wi_[ent] and compute 

**==> picture [371 x 28] intentionally omitted <==**

6 

where the entropy is normalized within the sequence so the weights span a broad dynamic range regardless of the absolute entropy scale. This attenuates low-entropy tokens instead of letting them dilute the score, while preserving the strongest evidence from uncertain positions. Unlike prior entropyfiltering approaches (Lee et al., 2023) that threshold the entropies, our scheme is continuous: every token still contributes, but with strength matched to its expected usefulness. Since the null statistic is a weighted sum of independent exponentials, the moment-matched Gamma approximation below provides calibrated _p_ -values that explicitly account for these entropy weights. 

_Moment-Matched Gamma Approximation. S_ combined is a weighted sum of independent, non-identical exponentials, which follows a hypoexponential whose CDF, while closed-form, is numerically unstable when rates are similar and costly to evaluate for large _n_ .[1] A Gaussian approximation fails to capture the heavy-tailed scores, so we use moment matching instead. Under _H_ 0, each term has mean _wi_[ent] and variance ( _wi_[ent] )[2] _θR_ . We fit _S_ combined _∼_ Gamma( _k_ new _, θ_ new) by matching the first two moments: 

**==> picture [324 x 26] intentionally omitted <==**

The resulting _p_ -value is: 

**==> picture [331 x 11] intentionally omitted <==**

We show in subsection 5.2 that this approximation is well-calibrated under _H_ 0, and in subsection 5.1 that it significantly outperforms unweighted detection. This framework handles dual-key routing ( _α_ ) and entropy gating ( _wi_[ent] ) in a single frequentist test. 

## **3.3 Multi-Region Localization and Adaptive Ensemble** 

When a document contains multiple scattered watermarked regions (e.g., distinct AI-generated paragraphs pasted into a human-written essay), evaluating a global score suffers from two critical flaws. First, the unwatermarked background tokens severely dilute the statistical signal. Second, it fails to identify the specific provenance of individual segments, which is critical for practical attribution. 

**Geometric Cover Search & Greedy Extraction.** To solve this, our goal is to extract a set of disjoint watermarked intervals _{_ [ _a_ 1 _, b_ 1] _, . . . ,_ [ _ay, by_ ] _}_ . A naive search over all _O_ ( _n_[2] ) possible start and end pairs is computationally prohibitive and incurs a massive multiple-testing penalty. Instead, we employ a geometric cover search, reducing the space to dyadic window lengths _L ∈{L_ 0 _,_ 2 _L_ 0 _,_ 4 _L_ 0 _, . . . ,_ 2 _[⌊]_[log][2] _[ n][⌋] }_ , sliding each window across the text at half-length offsets. This yields a strictly bounded number of candidate windows, _M ≈_ 4 _n/L_ min. 

The extraction proceeds in two stages. First, we rank all _M_ windows by their raw score sum (computed in _O_ (1) per window via prefix sums). Then, for the top candidates only, we compute the rigorous entropy-weighted Gamma _p_ -value. The greedy extraction selects the window with the lowest _p_ -value, flags it as watermarked, masks its tokens, and repeats on the residual text, aggregating intervals as long as their combined significance overcomes the multiple-testing tax. This localized extraction is governed by the minimum zone length _L_ min (default 50) and the maximum number of zones _Y_ max (default 5). Full mathematical details are provided in Appendix D. 

**Adaptive Ensemble Detection.** Discovering _y_ regions among _M_ candidates incurs a combinatorial multiple-testing tax. To adapt to any editing behavior, our ensemble selects the most significant among three strategies, applying a flat Bonferroni correction ( _k_ = 3): (1) **Global** full-text test (no search penalty), (2) **Single-Best** window (penalized by log10 _M_ ), and (3) **Multi-Region** aggregation over _M y_ zones (penalized by log10 � _y_ � + log10 _Y_ max). The final significance score is: 

log10 _p_ final = min� log10 _p_ global _,_ log10 _p_ single _,_ log10 _p_ multi� + log10 3 _._ (7) 

> 1The hypoexponential CDF involves a sum of _n_ exponentials with coefficients[�] _j_ = _i[λ][j][/]_[(] _[λ][j][−][λ][i]_[)][,][which][become] unstable when rates are close. In contrast, the Gamma CDF reduces to the well-optimized incomplete gamma function. 

7 

**The Dilution Rescue Effect.** For largely unedited text, the ensemble gracefully defaults to the global test, paying a negligible worst-case penalty of log10(3). Consider _w_ watermarked tokens with expected per-token score _µ >_ 1 (as given by Proposition 3), split into _y_ chunks within a document of length _n_ . Under extreme dilution ( _n ≫ w_ ), the global test’s significance drops as its signal-to-noise ratio scales by _O_ ( _w_[2] ( _µ −_ 1)[2] _/n_ ). The multi-region strategy isolates the pure signal ( _O_ ( _w_ ( _µ −_ 1)[2] )) but pays a combinatorial tax scaling as _O_ ( _y_ log10 _n_ ). Localization rescues detection when the isolated signal outpaces this logarithmic tax: _w_ ( _µ −_ 1)[2] ≳ _y_ log10 _n_ . For instance, _w_ = 800 tokens ( _µ_ = 1 _._ 2) in _y_ = 5 chunks easily overcome the 5 log10 _n_ tax, allowing confident detection even within _n_ = 100 _,_ 000 tokens—a scenario where the global signal is destroyed. 

**High-Resolution Boundary Annotation (mIoU).** While the greedy ensemble rigorously bounds the False Positive Rate, the harsh combinatorial tax forces it to prematurely discard small fragments, making it suboptimal for exact boundary estimation (mean Intersection over Union, or mIoU). To achieve highresolution localization, we decouple _detection_ from _annotation_ . If the ensemble definitively rejects the null hypothesis _H_ 0, we drop the search taxes and apply a localized density smoother. Tokens satisfying a normalized weighted moving average _S_[¯] _i > τ_ are locally annotated as watermarked, allowing the recovery of fine-grained, sentence-level provenance (see Appendix D for exact formulation). 

## **4 Main Experiments** 

## **4.1 Experimental Setup** 

_Models & Datasets_ Unless stated otherwise, we use Qwen 3.5-27B (Qwen Team, 2026) for generation, with _T_ = 0 _._ 8, top- _p_ = 0 _._ 9, and reasoning disabled, and we use entropy-weighted detection (subsection 3.2) with the more lightweight Qwen 3.5-0.8B model. We evaluate on 1k prompts from the ELI5 dataset (Fan et al., 2019) (with 5 different seeds), truncating answers to 400 tokens. 

We compare _TextSeal_ (default mixing parameter _α_ = 0 _._ 1 from subsection 3.1) to _Gumbel-Max_ (Aaronson and Kirchner, 2023) and _SynthID-Text_ (Dathathri et al., 2024) with depth 10. _SynthID-Text_ embeds a watermark via multi-layered tournament sampling with binary random functions, and proposes two detection methods: (i) a frequentist Z-test over the mean tournament score, and (ii) a Bayesian detector that estimates the posterior P(watermarked _|_ scores) via a logistic regression or MLP trained on a representative dataset. We use the frequentist Z-test, because the Bayesian detector provides no controlled false-positive rate, does not generalize across domains (its posteriors depend on the training distribution), and is incompatible with localized multi-window testing (full discussion in Appendix A.2). We fix the watermark context window size to _k_ = 3 for all methods, meaning the pseudo-random function depends on the three preceding tokens. At detection time, we deduplicate (context window, token) tuples, because the PRF is deterministic and repeated tuples would yield identical scores, violating the independence assumption of the statistical test (Fernandez et al., 2023). 

## **4.2 Detectability-Diversity Trade-off** 

A practical watermark must embed a robust signal without changing the output distribution. For Figure 1a, we vary for _TextSeal_ the mixing parameter _α_ from subsection 3.1 from 0 (deterministic) to 0.5 (blue and green curves). For _SynthID_ , we vary the depth from 2 to 20. _TextSeal_ consistently dominates the detectability–diversity trade-off. Furthermore, using the 27B model for entropy detection (“TextSeal high”) boosts detectability by 1–2 orders of magnitude at higher detection cost. 

## **4.3 Performance on Benchmarks** 

We evaluate how TextSeal’s watermarking impacts performance across a suite of 12 benchmarks spanning math, code, general knowledge, and common sense domains. We use Qwen 3.5-27B with _T_ = 0 _._ 6 (mild watermarking) and _T_ = 1 _._ 0 (stronger watermarking[2] ) and compare against vanilla generation without watermarking. Each benchmark is evaluated with generation at top- _p_ = 0 _._ 95, reasoning temperatures 0 _._ 6 or 1 _._ 0 and a maximum reasoning budget of 3,000 tokens. 

> 2The temperature controls the strength of the watermark since an increased temperature leads to higher entropy. 

8 

**Table 1** Accuracy across multiple benchmarks with and without _TextSeal_ (SQA[*] = SimpleQA). No significant performance drop is observed across benchmarks, confirming that TextSeal preserves the capabilities of the underlying model. 

|Reasoning temp. WM|Math<br> AIME MATH GSM8K Avg|Code<br>HE<br>MBPP Avg|Knowledge<br> MMLU GPQA SQA* Avg|Common Sense<br>HS<br>WG ARC-E ARC-C Avg<br>Avg|
|---|---|---|---|---|
|0.6<br>✓<br>✗|41.1<br>79.8<br>96.0<br>72.3 <br>40.1<br>79.8<br>95.4<br>71.7|93.3<br>49.2<br>71.2<br> 97.0<br>50.2<br>73.6|51.5<br>50.0<br>16.0<br>39.2 <br>49.2<br>50.5<br>15.8<br>38.5|94.8 93.5<br>93.7<br>88.5<br>92.6 **70.6**<br> 94.7 93.2<br>93.4<br>88.3<br>92.4 **70.6**|
|1.0<br>✓<br>✗|37.1<br>77.9<br>96.1<br>70.4 <br>35.8<br>78.4<br>96.1<br>70.1|94.5<br>48.5<br>71.5<br> 98.2<br>49.3<br>73.7|48.9<br>45.5<br>13.7<br>36.0 <br>46.4<br>42.9<br>15.5<br>34.9|94.6 93.8<br>92.8<br>86.0<br>91.8<br>**69.1**<br> 94.6 93.6<br>93.8<br>86.6<br>92.2 **69.3**|



**Table 2** Human preference evaluation (majority vote aggregation over 3 annotators per sample). Net Win Rate: ( _n_ WM _− n_ Base) _/N_ . _p_ -value: two-sided binomial test on decisive samples against 50%. No test reaches significance after Bonferroni correction ( _α/_ 6 = 0 _._ 008). 

|Language|WM|Wins|Base|Wins|Ties|WM|Preference|Rate|_p_-value|Net|Win Rate|
|---|---|---|---|---|---|---|---|---|---|---|---|
|English||124||146|1,730||45.9%||0.20||_−_1_._1%|
|Arabic||201||181|618||52.6%||0.33||+2_._0%|
|Chinese||84||74|842||53.2%||0.47||+1_._0%|
|Hindi||91||89|820||50.6%||0.94||+0_._2%|
|Japanese||143||130|727||52.4%||0.47||+1_._3%|
|Overall||643||620|4,737||50.9%||0.54||+0_._4%|



On average, TextSeal preserves the performance of the underlying model across benchmarks and temperature settings, with no significant differences. However, we observe a slight performance drop on code benchmarks (Human-eval: HE and MBPP) of 1-2 points. Analyzing the outputs suggests that this drop comes from minor formatting omissions rather than incorrect reasoning or algorithmic failures. In particular, all watermarked generations from HE that fail with watermarking and not without fail because they give only the function definition while still using annotations such as List[...], without adding the required from typing import List import. We note that benchmark evaluation inherently involves noise from the stochastic generation. To quantify this variance, we re-ran a subset of benchmarks with multiple random seeds and secret keys. The observed differences between watermarked and non-watermarked conditions fall within the variance introduced by seed/key changes, confirming that watermarking does not systematically degrade or improve performance (see subsection E.1). 

## **4.4 Human Evaluation of Imperceptibility** 

We assess whether the watermark introduces perceptible quality degradation through a human A/B preference study. Following the methodology of Dathathri et al. (2024), we generate paired responses to questions from ELI5 (Fan et al., 2019) (2,000 English samples) and CaLMQA (Arora et al., 2025) (1,000 each for Arabic, Chinese, Hindi, Japanese), totaling 6,000 question-answer pairs. 

Each pair is evaluated by three annotators (via Appen) with qualifications requiring post-graduate education, native-level language fluency, and at least two years of experience. Annotators select among four options: _A is preferred_ , _B is preferred_ , _both equally good_ , or _both equally bad_ , without knowing which output is watermarked. We aggregate via majority vote, merging the two tie categories and defaulting split votes (one vote per category) to tie. 

_Results._ Table 2 reports preference rates after majority vote aggregation. We test whether the watermark win rate among decisive (non-tie) samples differs from 50% using a two-sided binomial test. No individual language reaches significance (all _p >_ 0 _._ 05), and no test is significant after Bonferroni correction for the six comparisons ( _α/_ 6 = 0 _._ 008). The majority of samples (79%) result in ties, and inter-annotator agreement is high (88% of samples have at least 2/3 consensus on the 4-class scale). We also report the _net win rate_ , defined as ( _n_ WM _− n_ Base) _/N_ , with _n_ WM and _n_ Base the counts of samples where the watermarked or baseline response is preferred, and _N_ the total number of samples. The overall net win rate is +0 _._ 38%, indicating a negligible difference. 

To rigorously establish imperceptibility rather than failing to detect a difference, we apply the Two One- 

9 

Sided Tests (TOST) procedure (Schuirmann, 1987) for equivalence testing. We test _|P_ (WM preferred) _− P_ (Base preferred) _| <_ ∆ over all _N_ samples (including ties), which provides greater statistical power than restricting to decisive samples alone. With a smallest effect size of interest ∆= 5%, equivalence is established overall and for four of five individual languages ( _p <_ 0 _._ 05); Arabic marginally fails ( _p_ = 0 _._ 06) due to its wider confidence interval. The overall 90% CI is [ _−_ 0 _._ 6% _,_ +1 _._ 4%] _⊂_ [ _−_ 5% _,_ +5%]. Full breakdowns and the equivalence testing methodology are provided in Appendix E.3. 

## **4.5 Localization in Mixed Documents** 

In practice, watermarked text often forms only a fraction of a document (e.g., AI-generated paragraphs within a human-written report). A global detector scoring the entire text faces two primary challenges: _dilution_ , where unwatermarked tokens degrade the signal-to-noise ratio, and _fragmentation_ , where watermarked content is scattered across non-contiguous regions. We evaluate TextSeal’s adaptive ensemble (Section 3.3) against global detection by embedding 400-token watermarked answers ( _T_ = 1 _._ 0, top- _p_ = 0 _._ 95, chosen to increase the watermark signal) into unwatermarked Wikipedia texts. Under **dilution (** _K_ =1 **)** , we place a single contiguous 400-token block inside documents of increasing length, up to 12 _,_ 000 tokens (watermarked fraction: 3 _._ 3%). Under **fragmentation (** _K>_ 1 **)** , we split the 400 tokens into _K ∈{_ 1 _,_ 2 _,_ 3 _,_ 5 _}_ equal fragments interleaved within a fixed 8 _,_ 000-token document. 

_Results._ As shown in Figure 3 (left), global detection suffers heavily from dilution, degrading at roughly _O_ (1 _/T_ ) and failing to reach significance ( _p_ =0 _._ 01) beyond _T_ =4000. TextSeal’s adaptive ensemble, however, efficiently isolates the signal, maintaining strong detectability ( _−_ log10 _p >_ 4) even at _T_ =12 _,_ 000 (a 30 _×_ dilution). For fragmentation (Figure 3, right), global detectors exhibit flat performance, as they are blind to spatial arrangement. Conversely, the ensemble successfully detects the watermark for up to _K_ =3 fragments. Performance only degrades at _K_ =5, where individual fragments ( _∼_ 80 tokens) become too small to overcome the statistical penalty of multiple-hypothesis testing. Overall, TextSeal’s localized approach dramatically outperforms global baselines whenever watermarked content is reasonably concentrated within the document. 

**==> picture [442 x 176] intentionally omitted <==**

**----- Start of picture text -----**<br>
TextSeal (ensemble) TextSeal (ensemble)<br>TextSeal (global) 4.0 TextSeal (global)<br>8 SynthID SynthID<br>3.5<br>6 3.0<br>2.5<br>p = 0.01<br>4 2.0<br>1.5<br>2 p = 0.01 1.0<br>0.5<br>0 0.0<br>0 2 4 6 8 10 12 1 2 3 5<br>Document length (k tokens) Fragments (K)<br>)  )<br> p  p<br>10 10<br>log log<br>Detectability ( Detectability (<br>**----- End of picture text -----**<br>


**Figure 3** Localized detection in mixed documents containing 400 watermarked tokens. **(Left)** _Dilution_ : A single watermarked block ( _K_ =1) embedded in documents of increasing length. Global detection (light blue and red curves) degrades rapidly as the watermarked fraction shrinks, dropping below the _p_ =0 _._ 01 significance threshold around 4k tokens. The adaptive ensemble (dark blue curve) maintains strong detectability ( _−_ log10 _p >_ 4) even at 12k tokens (3 _._ 3% watermarked). **(Right)** _Fragmentation_ : Watermarked text split into _K_ fragments within an 8000-token document. Global detectors are insensitive to fragmentation (flat curves at _−_ log10 _p ≈_ 1), while the ensemble leverages localized search to extract the signal at _K≤_ 3 fragments. 

10 

## **5 Ablations and Analyses** 

## **5.1 Diversity Strategies Comparison** 

We compare four strategies for restoring diversity in Gumbel-max watermarking (full descriptions and proofs in Appendix C): **(1)StochasticMixing** mixes the PRF value with a Bernoulli coin (control: mixing rate _a_ ); **(2) Periodic Skip** disables watermarking at fixed intervals (control: skip rate _α_ ); **(3) EntropyNormalized Skip** skips watermarking with probability _τ_ uniformly across entropy regimes, preserving distortion-freeness (control: target skip rate _τ_ ); **(4)Dual-KeyRouting** (subsection 3.1) alternates between two secret keys (control: routing probability _α ∈_ [0 _,_ 0 _._ 5]). 

_Results: Diversity vs. Detectability Trade-off._ We evaluate Qwen 3.5-27B on 1k ELI5 prompts, with reasoning disabled, temperature 1 _._ 0, top- _p_ = 0 _._ 95, maximum generation length 2,048, watermark context size _k_ = 3, and two generations per prompt to compute Self-BLEU. For detection, we report both the classical test and the entropy-weighted Gamma test. For each method, we vary the control hyperparameter to trace the Pareto frontier between _diversity_ (measured by Self-BLEU, where lower indicates more diverse) and _detectability_ ( _p_ -value under _H_ 0; lower means stronger watermark). Figure 4 illustrates the Pareto frontiers for all five methods. Ideally, a method should push towards the top-left corner (low Self-BLEU, low _p_ -value). Several trends stand out. First, entropy-weighted detection consistently improves every method, often by several orders of magnitude in median _p_ -value, without changing the generation diversity. Second, early-fusion dual-key routing clearly outperforms Fisher-style dual-key aggregation at comparable Self-BLEU, confirming that early fusion is the right detector for routed generation. Third, stochastic mixing is consistently dominated: it reaches similar or worse detectability only at much higher Self-BLEU, making it a poor trade-off in practice. 

Among the strongest methods, entropy skip and early-fusion dual-key routing define the best Pareto frontier. Entropy skip is slightly stronger at the highest-detectability end, while early-fusion dual-key routing remains very close across the full sweep and has the practical advantage of mapping directly to speculative decoding and MTP-style deployments. We therefore select dual-key routing as the default diversity mechanism for TextSeal in all experiments. 

## **5.2 False Positive Rate Check** 

A reliable detector must strictly control its empirical False Positive Rate (FPR) at any nominal threshold _τ_ . We validate this on 1 million unwatermarked Wikipedia passages (256 tokens each), rather than ELI5 answers to have more texts and cover a wider distribution. We plot in Figure 5 the empirical FPR against _τ_ ; perfect calibration aligns with the diagonal, while curves _above_ it indicate safe, conservative behavior. Under standard unweighted dual-key detection presented in subsection 3.1 (Figure 5, left), 

**==> picture [420 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
12<br>10<br>8<br>6<br>Classical<br>4 Entropy weighted<br>Dual-key (Early fusion)<br>Dual-key (Fisher)<br>2 Mix<br>Skip<br>Entropy Skip<br>0<br>0.150 0.175 0.200 0.225 0.250 0.275 0.300 0.325<br>Self-BLEU (Diversity,  )<br>)<br>-value) (Detectability,<br>p<br> (<br>10<br>log<br>**----- End of picture text -----**<br>


**Figure 4 Pareto frontier of diversity strategies.** Self-BLEU is lower-is-better and median _−_ log10( _p_ ) is higher-isbetter. Solid lines use the classical detector; dashed lines use entropy weighting. Early-fusion dual-key routing outperforms Fisher at matched diversity, and together with entropy skip defines the strongest frontier. 

11 

**Table 3** Quality comparison between watermarked (WM) and non-watermarked (Non-WM) answers on 6,000 multilingual QA pairs. Reasoning/Answer tokens: average per response. Refusal/Script: percentage of responses with refusal or wrong language script. Results show no meaningful quality difference between conditions. 

|Language|Reasoning tokens<br>WM<br>Non-WM|Answer tokens<br>WM<br>Non-WM<br>|Refusal %<br>W<br>WM<br>Non-WM<br>W|rong Script %<br>M<br>Non-WM|
|---|---|---|---|---|
|English<br>Arabic<br>Chinese<br>Hindi<br>Japanese|168<br>149<br>296<br>232<br>224<br>203<br>293<br>229<br>294<br>280|124<br>121<br>161<br>155<br>144<br>146<br>166<br>161<br>181<br>181|0.8<br>0.7<br><br>0.6<br>0.6<br><br>0.6<br>0.3<br><br>1.1<br>1.1<br><br>0.6<br>0.2<br>|0.0<br>0.0<br>0.6<br>0.6<br>0.7<br>0.6<br>1.1<br>1.1<br>9.0<br>7.8|
|Overall|240<br>207|150<br>147|0.7<br>0.6<br>|1.9<br>1.7|



all methods tightly track the diagonal down to _τ ≈_ 10 _[−]_[4] . Under lightweight (0.8B) entropy-weighted detection as described in subsection 3.2 (Figure 5, right), TextSeal remains strictly well-calibrated. 

## **5.3 Generalization: Multilingual Question Answering** 

The experiments above use a single model (Qwen 3.5-27B) on English text. To test whether TextSeal generalizes across models, languages, and scripts, we evaluate on a multilingual question-answering task using a different model: GPT-OSS-20B, OpenAI’s open-weights 20B-parameter reasoning model, with reasoning enabled, on two datasets: ELI5 (English, 2,000 questions) and CalmQA (Arabic, Chinese, Hindi, Japanese; 1,000 questions each), totaling 6,000 paired samples. Each question is answered with and without watermarking (top- _p_ = 0 _._ 95, temperature= 0 _._ 7). Full experimental details are in subsection E.2. 

_Quality Comparison._ Table 3 compares generation quality between watermarked and non-watermarked outputs. Reasoning lengths show a small increase with watermarking ( _∼_ 16% more reasoning tokens, especially in other languages than English), likely due to sampling variance rather than a systematic effect. Refusal rates are low under both conditions ( _<_ 1%). Script consistency is _>_ 98% for all languages except Japanese (90%), with a small increase of 1% for WM. We use McNemar’s tests (McNemar, 1947) to confirm that there is no statistically significant difference between conditions for either refusal rates ( _p_ = 0 _._ 41) or script consistency ( _p_ = 0 _._ 21); see subsection E.2 for details. TextSeal achieves 63.3% TPR at 0.1% FPR overall; per-language detection results are in App. E.2. 

**==> picture [442 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [0] 10 [0]<br>10 1 10 1<br>10 2 10 2<br>Ideal Ideal<br>SynthID d=10 SynthID d=10<br>10 3 SynthID d=20 10 3 SynthID d=20<br>TextSeal =0.0 TextSeal =0.0<br>10 4 TextSeal =0.1 TextSeal  =0.3 10 4 TextSeal =0.1 TextSeal  =0.3<br>TextSeal  =0.5 TextSeal  =0.5<br>10 5 10 5<br>10 5 10 4 10 3 10 2 10 1 10 [0] 10 5 10 4 10 3 10 2 10 1 10 [0]<br>Empirical FPR Empirical FPR<br>)<br>Theoretical FPR (<br>**----- End of picture text -----**<br>


**Figure 5** Theoretical FPR ( _τ_ ) vs. empirical FPR under standard detection (left) and entropy-weighted linear detection (right), on 1M unwatermarked Wikipedia texts (256 tokens each) The dashed diagonal indicates perfect calibration. All curves lie above the diagonal (conservative): the empirical FPR never exceeds the nominal level. Line styles distinguish parameter settings within each method. 

12 

**Table 4** Per-token sampling overhead of TextSeal and SynthID watermarking on a single H200 GPU. Each method is measured on the _same_ logits, isolating the sampling cost. TextSeal uses dual-key Gumbel-Max ( _α_ =0 _._ 1, _n_ -gram=3); SynthID uses tournament depth _d_ =10. Median over 30 prompts of ELI-5 _×_ 400 tokens. 

|**Model**<br>**Fwd**<br>(ms)|**No Watermark**<br>**Sample**<br>**tok/s**<br>(ms)|**TextSeal**<br>**Sample**<br>**Overhead**<br>(ms)|**SynthID**<br>**Sample**<br>**Overhead**<br>(ms)|
|---|---|---|---|
|Qwen 3.5-0.8B<br>21.4<br>Qwen 3.5-2B<br>21.5<br>Qwen 3.5-4B<br>30.3<br>Qwen 3.5-9B<br>31.3<br>Qwen 3.5-27B<br>61.9|0.37<br>45.9<br>0.36<br>45.8<br>0.38<br>32.6<br>0.38<br>31.5<br>0.39<br>16.1|0.43<br>0_._3%<br>0.43<br>0_._3%<br>0.44<br>0_._2%<br>0.45<br>0_._2%<br>0.46<br>0_._1%|0.61<br>1_._1%<br>0.60<br>1_._1%<br>0.62<br>0_._8%<br>0.62<br>0_._8%<br>0.63<br>0_._4%|



## **5.4 Real-World Considerations: Embedding and Detection Efficiency** 

We evaluate TextSeal’s computational efficiency during both generation and detection to ensure it is lightweight enough for large-scale deployment. 

_Generation Overhead._ Table 4 details the sampling overhead during autoregressive decoding. TextSeal evaluates a fused dual-key pseudorandom function (PRF) restricted strictly to the top- _p_ survivor tokens ( _∼_ 200 tokens), avoiding full-vocabulary hashes. This adds only _∼_ 0 _._ 07 ms per token ( _≤_ 0 _._ 3% overhead). In contrast, SynthID’s iterative tournament sampling requires _d_ sequential rounds of top- _p_ reweightings and a multinomial sampling step, costing _∼_ 0 _._ 6 ms per token (0 _._ 4–1 _._ 1% overhead). Crucially, both methods operate entirely on the logits, requiring no model parameter changes or KV-cache modifications, ensuring immediate compatibility with standard serving infrastructure. 

_Detection Efficiency and Proxy Scaling._ For detection, we evaluate the optimal proxy model size for entropy weighting (Section 3.2) across varying attack strengths (Figure 6). Standard unweighted detection is highly efficient (0 _._ 007 ms/token, 0 GB VRAM overhead) but yields a baseline median _−_ log10 _p_ of 4 _._ 8. Entropy weighting with the full 27B model significantly boosts this score to 8 _._ 2, but incurs massive overhead (50 _._ 1 GB VRAM, 0 _._ 213 ms/token). However, the 4-bit quantized 0.8B model emerges as the optimal practical choice: it achieves near-parity detectability (6 _._ 2) and scales identically against robust attacks, while requiring only 1 _._ 4 GB VRAM and 0 _._ 115 ms/token. 

_MTP Speculative Decoding._ Multi-token prediction (MTP) speculative decoding accelerates inference via lightweight draft heads that propose multiple tokens in parallel (Gloeckle et al., 2024). TextSeal natively supports this by assigning key _A_ to draft-accepted tokens and key _B_ to target-resampled tokens. 

**==> picture [354 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
8 Standard 10 [2] 50.1<br>Qwen 0.8BQwen 9B 0.213 16.7<br>6 Qwen 27B 10 1 0.083 0.109 10 [1]<br>1.4<br>10 [0]<br>4<br>2 p = 0.01 10 2 0.007 10 1<br>0<br>0 10 2<br>None Copy Light Select. Rephr.Rewrite Standard 0.8B 9B 27B Standard 0.8B 9B 27B<br>edit rephr. rephr. Entropy model Entropy model<br>Attack strength<br>)<br>p<br>10<br>log GPU memory (GB)<br>Detection time (ms / token)<br>Detectability (med.<br>**----- End of picture text -----**<br>


**Figure 6** Entropy-aware detection performance and computational costs. **(Left)** Detectability under varying attack strengths. The highly efficient 4-bit 0.8B model boosts base detectability by _∼_ 1 _._ 3 orders of magnitude, capturing much of the theoretical maximum boost (+3 _._ 4) provided by the 27B generation model. **(Middle)** Detection time per token (log scale). **(Right)** Peak GPU memory allocation (log scale). The 4-bit 0.8B model offers an excellent trade-off, recovering most of the watermark signal while requiring 35 _×_ less memory. 

13 

Consequently, the mixing parameter _α_ dynamically matches the empirical acceptance rate. We evaluate Qwen 3.5 (2B, 9B, 27B) generating 400-token ELI5 answers under three conditions: standard MTP, MTP with TextSeal, and autoregressive TextSeal. As shown in Figure 7, MTP draft acceptance rates remain identical (29–46%) with and without TextSeal, confirming the dual-key approach is perfectly distortion-free and preserves all speculative efficiency gains. While MTP TextSeal’s detection signal is slightly lower than standard TextSeal due to key mixing dilution ( _α <_ 1), entropy weighting easily recovers strong significance well above the _p_ =0 _._ 01 threshold. Perplexity remains identical across all conditions and model sizes, confirming that the dual-key watermark introduces no quality degradation. 

## **6 Watermark Radioactivity: Detecting Distillation via Learnability** 

A watermark is _radioactive_ (Sander et al., 2024) if, when a model is trained on watermarked data, it inherits a detectable token bias. This enables a powerful application beyond text provenance: detecting whether a competitor has distilled your model’s outputs into their own. 

_Setup._ We distill DeepSeek-R1-Distill-Qwen-14B (Guo et al., 2025) (teacher) into Qwen2.5-3B (Team, 2024) (student) on 5,000 curated problems from OpenR1-Math-220k (Hugging Face, 2025). The teacher generates reasoning traces under four watermark schemes: Gumbel-Max (Aaronson and Kirchner, 2023), TextSeal ( _α_ =0 _._ 1), SynthID depth 10 (Dathathri et al., 2024), and an unwatermarked control (all with watermark windows of size 3). Following Muennighoff et al. (2025), we retain traces only if they close their </think> block, contain a \boxed{} answer (when required), match the reference solution under math_verify, and have no 100-character span recurring _≥_ 3 times. We also remove problems that the base student already solves correctly, so the distillation set only includes traces that teach the student something new. We then apply LoRA fine-tuning on the remaining traces. 

_Detection methodology._ To test whether the watermark transferred, we use the open-model radioactivity test (Sander et al., 2024). We feed each training trace into the student using _teacher forcing_ (providing the ground-truth prefix at each position) and record the student’s top-1 prediction. If the student internalized the watermark bias during training, its predictions should be skewed toward highPRF tokens. We score each prediction with the watermark PRF and aggregate into a _p_ -value. To get a statistically valid test, we deduplicate at two levels. _Within each trace,_ each watermark context window **w** _t_ is scored at most once: if the same _k_ -gram appears more than once in a trace, we score the student’s prediction only at the first occurrence. This is needed to avoid spurious signal: a high-PRF token already inside the input context can be copied by the student through attention rather than retrieved from internalized watermark bias. _Across traces,_ we further deduplicate _(context window, predicted token)_ tuples globally so that repeated tuples are counted only once. The PRF is deterministic in ( _v,_ **w** _, K_ ), so duplicated tuples produce identical scores and would violate independence in the 

**==> picture [354 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.5<br>MTP 20 TextSeal (ent.) TextSeal<br>50% MTP TextSeal TextSeal (std) MTP TextSeal<br>MTP TextSeal (ent.) 2.0 MTP<br>40% 15 MTP TextSeal (std)<br>1.5<br>30%<br>10<br>1.0<br>20%<br>10% 5 p = 0.01 0.5<br>0% 0 0.0<br>2B 9B 27B 2B 9B 27B 2B 9B 27B<br>Model size Model size Model size<br>)<br> p<br>10<br>log<br>Perplexity<br>Acceptance rate<br>Detectability (<br>**----- End of picture text -----**<br>


**Figure 7** MTP speculative decoding with TextSeal watermarking across Qwen 3.5 model sizes (2B, 9B, 27B) at temperature 0 _._ 8, top- _p_ 0 _._ 9. _Left:_ Draft acceptance rate is unchanged by dual-key watermarking, confirming zero overhead. _Center:_ Both TextSeal and MTP TextSeal are well above the _p_ =0 _._ 01 detection threshold; solid bars show entropy-weighted detection, hatched bars show standard detection. The modest gap between TextSeal and MTP TextSeal is explained by the key-mixing parameter _α <_ 1. _Right:_ Perplexity is identical across all conditions, confirming that watermarking is distortion-free. 

14 

**Table 5 Teacher trace quality and detectability.** The teacher generates 5,000 traces per method; pass rate is the fraction retained by the four-stage quality filter. Teacher _−_ log10( _p_ ) reports the mean watermark detection power across individual teacher traces. Accuracy is measured on GSM8K (1,319 problems, greedy decoding); the baseline is the pre-training Qwen2.5-3B.[†] TextSeal uses entropy-weighted scoring. 

|**Method**|**Retained**|**Pass**|**Teacher**|**GSM8K**|∆**vs**|
|---|---|---|---|---|---|
||**Traces**|**Rate**|_−_log10(_p_)|**Acc**|**Base**|
|Base Model (Qwen2.5-3B)|—|—|—|64.5%|—|
|Gumbel-Max|1,991|39.8%|14.89|78.8%|+14.3|
|TextSeal|2,352|47.0%|33.15†|79.9%|+15.4|
|SynthID|2,408|48.2%|14.39|75.2%|+10.7|
|Control|2,400|48.0%|0.39|75.5%|+11.0|



statistical test. After deduplication, this yields _∼_ 1 _._ 4–2 _._ 2M unique scored tokens per method (full setup in subsection E.4). 

_Results._ Figure 8 shows that all three watermarks reliably transfer through distillation, with detection power far exceeding the significance threshold. Under the original setup (each method uses all its retained traces), TextSeal achieves the strongest signal thanks to higher data volume. Once data volume is equalized (controlled conditions in Figure 8b,c), Gumbel-Max dominates, confirming a stronger per-token signal via deterministic argmax; TextSeal achieves comparable overall detectability by retaining more training data. All distilled students substantially improve over the base model (+10–15% on GSM8K), and distilling on watermarked traces does not lead to significant changes compared to the unwatermarked control. 

_Controlled comparisons._ To rule out training data volume as a confound, we repeat the experiment under two controlled conditions: (i) _equal traces_ , where each method uses exactly 1 _,_ 991 traces (the Gumbel-Max minimum, randomly subsampled for the other methods); and (ii) _equal tokens_ , where each method is allocated _∼_ 15 _._ 1M characters. Under equal traces, TextSeal achieves the highest student accuracy (81 _._ 0%), followed by SynthID and Control (78 _._ 8% each) and Gumbel-Max (77 _._ 7%). Under equal tokens, the spread narrows (79 _._ 7%/78 _._ 6%/79 _._ 6%/77 _._ 6% for TextSeal/Gumbel-Max/SynthID/Control). Detection remains strong under both controls, validating that the conclusions of Figure 8 are not artifacts of unequal training data volume. 

_Entropy weighting ablation._ For TextSeal we use � _H_ ˆ entropy-aware scoring by default (subsection 3.2). Figure 9 compares eight weighting functions in the same teacher-forcing setup, spanning normalized-entropy transforms (Sqrt, Log, Linear, Tanh of _H_[ˆ] _i_ ) and raw entropy power functions ( _Hi[β]_ ˆ for _β ∈{_ 0 _._ 5 _,_ 1 _._ 0 _,_ 1 _._ 5 _}_ ). The concave � _H_ weighting achieves the strongest detection ( _p_ = 3 _._ 7 _×_ 10 _[−]_[110] ), 

**==> picture [442 x 124] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Original (b) Equal Traces (c) Equal Tokens<br>35 GumbelMax 25 GumbelMax 25 GumbelMax<br>TextSeal TextSeal TextSeal<br>30 SynthID SynthID SynthID<br>Control 20 Control 20 Control<br>25<br>20 15 15<br>15 10 10<br>10<br>5 5<br>5<br>= 0.05 = 0.05 = 0.05<br>0 0 0<br>10 [4] 10 [5] 10 [6] 10 [4] 10 [5] 10 [6] 10 [4] 10 [5] 10 [6]<br>Scored Tokens Scored Tokens Scored Tokens<br>)p<br> (<br>10<br>log<br>**----- End of picture text -----**<br>


**Figure8 Watermarkradioactivitythroughdistillation.** Detection power ( _−_ log10( _p_ )) vs. number of unique scored tokens under three conditions: original traces, equal-trace control (1 _,_ 991 each), and equal-token control ( _∼_ 15 _._ 1M chars each). TextSeal achieves the strongest signal under the original setup thanks to retaining more traces, while Gumbel-Max dominates under controlled conditions, confirming its stronger per-token signal. 

15 

**==> picture [266 x 169] intentionally omitted <==**

**----- Start of picture text -----**<br>
Uniform (baseline) Tanh<br>10 SqrtLog HH 0.51.0 Best (Sqrt): p = 1.0e 09<br>Linear H 1.5<br>8<br>Baseline: p = 2.9e 08<br>6<br>4<br>p = 0.001<br>2<br>p = 0.05<br>0<br>1K 10K 100K 1.0M<br>Scored tokens<br>)p<br> (<br>10<br>log<br>**----- End of picture text -----**<br>


**Figure 9 Entropy-aware scoring for watermark learnability detection.** Detection power ( _−_ log10( _p_ )) vs. number of unique scored tokens in the teacher-forcing radioactivity test for TextSeal ( _α_ =0 _._ 1), comparing different entropy ˆ weighting functions _wi_[ent] = _f_ ( _Hi_ ) against a uniform (unweighted) baseline. The concave ~~�~~ _H_ weighting achieves the strongest signal ( _p_ = 3 _._ 7 _×_ 10 _[−]_[110] ), improving over the uniform baseline ( _p_ = 2 _._ 1 _×_ 10 _[−]_[84] ) by more than 25 orders of magnitude. 

improving over the uniform baseline ( _p_ = 2 _._ 1 _×_ 10 _[−]_[84] ) by more than 25 orders of magnitude. Concave functions outperform linear and superlinear alternatives because they moderately upweight highentropy positions—where the watermark has more room to influence token selection (Proposition 3)— without over-amplifying noisy extreme-entropy tokens. 

The full benchmark accuracy numbers and further details are given in subsection E.4. 

## **7 Conclusion** 

We introduced TextSeal, a distortion-free watermark for LLMs that achieves state-of-the-art detectability through dual-key generation, entropy-weighted detection, and localized multi-region search. TextSeal strictly dominates SynthID on the diversity-detectability frontier, preserves model performance across 12 benchmarks, supports speculative decoding and MTP, and transfers through distillation for radioactive tracing. 

_Limitations._ Like all distortion-free sampling watermarks, TextSeal trades diversity, not quality, for detectability. While this trade-off is invisible to users who observe a single generation, it may affect workflows that rely on diverse outputs (best-of- _N_ reranking, creative brainstorming). In practice, modern reasoning models trained with RL already exhibit collapsed entropy, limiting the marginal diversity loss; quantifying this across model families remains open. 

16 

## **References** 

- Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (AI Act), 2024. 

Scott Aaronson and Hendrik Kirchner. Watermarking GPT outputs, 2023. 

- Sahar Abdelnabi and Mario Fritz. Adversarial watermarking transformer: Towards tracing text provenance with data hiding. In _2021 IEEE Symposium on Security and Privacy (SP)_ , pages 121–140. IEEE, 2021. 

- Shane Arora, Marzena Karpinska, Hung-Ting Chen, Ipsita Bhattacharjee, Mohit Iyyer, and Eunsol Choi. Calmqa: Exploring culturally specific long-form question answering across 23 languages. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 11772–11817, 2025. 

- Guorui Bian, Michael McAleer, and Wing-Keung Wong. A trinomial test for paired data when there are many ties. _Mathematics and Computers in Simulation_ , 81(6):1153–1160, 2011. 

- Igor A Bolshakov. A method of linguistic steganography based on collocationally-verified synonymy. In _International Workshop on Information Hiding_ , pages 180–191. Springer, 2004. 

- Jack T Brassil, Steven Low, Nicholas F Maxemchuk, and Lawrence O’Gorman. Electronic marking and identification techniques to discourage document copying. _IEEE Journal on Selected Areas in Communications_ , 13(8):1495–1504, 1995. 

- Ching-Yun Chang and Stephen Clark. Practical linguistic steganography using contextual synonym substitution and a novel vertex coding method. _Computational linguistics_ , 40(2):403–448, 2014. 

- Mark Chapman, George I Davida, and Marc Rennhard. A practical and effective approach to large-scale automated linguistic steganography. In _International Conference on Information Security_ , pages 156–165. Springer, 2001. 

- Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. _Cryptology ePrint Archive_ , 2023. 

- Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, et al. Scalable watermarking for identifying large language model outputs. _Nature_ , 634(8035):818–823, 2024. 

- European Commission. Code of practice on marking and labelling of AI-generated content, 2026. Second draft published March 2026; enforcement of Article 50 obligations begins August 2, 2026. 

- Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form question answering. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 3558–3567. Association for Computational Linguistics, 2019. 

- Pierre Fernandez, Antoine Chaffin, Karim Tit, Vivien Chappelier, and Teddy Furon. Three bricks to consolidate watermarks for large language models. _2023 IEEE International Workshop on Information Forensics and Security (WIFS)_ , 2023. 

- Pierre Fernandez, Tom Sander, Hady Elsahar, Hongyan Chang, Tomáš Souček, Valeriu Lacatusu, Tuan Tran, Sylvestre-Alvise Rebuffi, and Alexandre Mourachko. How good is post-hoc watermarking with language model rephrasing? _arXiv preprint arXiv:2512.16904_ , 2025. 

- Yu Fu, Deyi Xiong, and Yue Dong. Watermarking conditional text generation for ai detection: Unveiling challenges and a semantic-aware watermark remedy. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pages 18003–18011, 2024. 

- Eva Giboulot and Teddy Furon. Watermax: breaking the llm watermark detectability-robustness-quality trade-off. _arXiv preprint arXiv:2403.04808_ , 2024. 

- Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. _arXiv preprint arXiv:2404.19737_ , 2024. 

- Chenchen Gu, Xiang Lisa Li, Percy Liang, and Tatsunori Hashimoto. On the learnability of watermarks for language models. _arXiv preprint arXiv:2312.04469_ , 2023. 

- Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_ , 2025. 

17 

- Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. _arXiv preprint arXiv:1904.09751_ , 2019. 

- Abe Bohan Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. _arXiv preprint arXiv:2310.03991_ , 2023. 

- Abe Bohan Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, and Tianxing He. k-semstamp: A clusteringbased semantic watermark for detection of machine-generated text. _arXiv preprint arXiv:2402.11399_ , 2024. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _Iclr_ , 1(2):3, 2022. 

- Baihe Huang, Banghua Zhu, Hanlin Zhu, Jason D. Lee, Jiantao Jiao, and Michael I. Jordan. Towards optimal statistical watermarking, 2023. 

- Hugging Face. Open r1: A fully open reproduction of deepseek-r1, 2025. 

- Nikola Jovanović, Robin Staab, Maximilian Baader, and Martin Vechev. Ward: Provable rag dataset inference via llm watermarks. _ICLR_ , 2025. 

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. _arXiv preprint arXiv:2301.10226_ , 2023a. 

- John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein. On the reliability of watermarks for large language models, 2023b. 

- Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. _arXiv preprint arXiv:2307.15593_ , 2023. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th symposium on operating systems principles_ , pages 611–626, 2023. 

- Gregory Kang Ruey Lau, Xinyuan Niu, Hieu Dao, Jiangwei Chen, Chuan-Sheng Foo, and Bryan Kian Hsiang Low. Waterfall: Framework for robust and scalable text watermarking. In _ICML 2024 Workshop on Foundation Models in the Wild_ , 2024. 

- Taehyun Lee, Seokhee Hong, Jaewoo Ahn, Ilgee Hong, Hwaran Lee, Sangdoo Yun, Jamin Shin, and Gunhee Kim. Who wrote this code? watermarking for code generation. _arXiv preprint arXiv:2305.15060_ , 2023. 

- Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In _ICML_ , 2023. 

- Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, and Lijie Wen. A semantic invariant robust watermark for large language models. _arXiv preprint arXiv:2310.06356_ , 2023. 

- Yepeng Liu and Yuheng Bu. Adaptive text watermark for large language models. _arXiv preprint arXiv:2401.13927_ , 2024. 

- Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. _Psychometrika_ , 12(2):153–157, 1947. 

- Hasan Mesut Meral, Bülent Sankur, A Sumru Özsoy, Tunga Güngör, and Emre Sevinç. Natural language watermarking via morphosyntactic alterations. _Computer Speech & Language_ , 23(1):107–125, 2009. 

- Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori B Hashimoto. s1: Simple test-time scaling. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 20286–20332, 2025. 

- Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, et al. Markllm: An open-source toolkit for llm watermarking. _arXiv preprint arXiv:2405.10051_ , 2024. 

- Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, and David Wagner. Mark my words: Analyzing and evaluating language model watermarks. _arXiv preprint arXiv:2312.00273_ , 2023. 

- Jipeng Qiang, Shiyu Zhu, Yun Li, Yi Zhu, Yunhao Yuan, and Xindong Wu. Natural language watermarking via paraphraser-based lexical substitution. _Artificial Intelligence_ , 317:103859, 2023. 

18 

- Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, and Jiaheng Zhang. Provably robust multi-bit watermarking for ai-generated text via error correction code. _arXiv preprint arXiv:2401.16820_ , 2024. 

- Qwen Team. Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5, 2026. Alibaba Cloud. 

- Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Hervé Jégou. Radioactive data: tracing through training. In _International Conference on Machine Learning_ , pages 8326–8335. PMLR, 2020. 

- Tom Sander, Pierre Fernandez, Alain Durmus, Matthijs Douze, and Teddy Furon. Watermarking makes language models radioactive. _NeurIPS_ , 2024. 

- Tom Sander, Pierre Fernandez, Saeed Mahloujifar, Alain Durmus, and Chuan Guo. Detecting benchmark contamination through watermarking. _arXiv preprint arXiv:2502.17259_ , 2025. 

- Donald J Schuirmann. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. _Journal of Pharmacokinetics and Biopharmaceutics_ , 15(6):657–680, 1987. 

- M Hassan Shirali-Shahreza and Mohammad Shirali-Shahreza. A new synonym text steganography. In _2008 international conference on intelligent information hiding and multimedia signal processing_ , pages 1524–1526. IEEE, 2008. 

Qwen Team. Qwen2.5 technical report. _arXiv preprint arXiv:2409.12117_ , 2024. 

- Mercan Topkara, Cuneyt M Taskiran, and Edward J Delp III. Natural language watermarking. In _Security, Steganography, and Watermarking of Multimedia Contents VII_ , pages 441–452. SPIE, 2005. 

- Mercan Topkara, Giuseppe Riccardi, Dilek Hakkani-Tür, and Mikhail J Atallah. Natural language watermarking: Challenges in building a practical system. In _Security, Steganography, and Watermarking of Multimedia Contents VIII_ , pages 106–117. SPIE, 2006a. 

- Mercan Topkara, Umut Topkara, and Mikhail J Atallah. Words are not enough: sentence level natural language watermarking. In _Proceedings of the 4th ACM international workshop on Contents protection and security_ , pages 37–46, 2006b. 

- Umut Topkara, Mercan Topkara, and Mikhail J Atallah. The hiding virtues of ambiguity: quantifiably resilient watermarking of natural language text through synonym substitutions. In _Proceedings of the 8th workshop on Multimedia and security_ , pages 164–174, 2006c. 

- Honai Ueoka, Yugo Murawaki, and Sadao Kurohashi. Frustratingly easy edit-based linguistic steganography with a masked language model. _arXiv preprint arXiv:2104.09833_ , 2021. 

- Ashish Venugopal, Jakob Uszkoreit, David Talbot, Franz Josef Och, and Juri Ganitkevitch. Watermarking the outputs of structured prediction with an application in statistical machine translation. In _Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing_ , pages 1363–1372, 2011. 

- Zongqi Wang, Tianle Gu, Baoyuan Wu, and Yujiu Yang. Morphmark: Flexible adaptive watermarking for large language models. _arXiv preprint arXiv:2505.11541_ , 2025. 

- Alex Wilson and Andrew D Ker. Avoiding detection on twitter: embedding strategies for linguistic steganography. _Electronic Imaging_ , 28:1–9, 2016. 

- Keith Winstein. Lexical steganography through adaptive modulation of the word choice hash. _Unpublished. http://www. imsa. edu/˜ keithw/tlex_ , 1998. 

- Yihan Wu, Zhengmian Hu, Hongyang Zhang, and Heng Huang. Dipmark: A stealthy, efficient and resilient watermark for large language models. _arXiv preprint arXiv:2310.07710_ , 2023. 

- Lingyun Xiang, Xinhui Wang, Chunfang Yang, and Peng Liu. A novel linguistic steganography based on synonym run-length encoding. _IEICE transactions on Information and Systems_ , 100(2):313–322, 2017. 

- Xiaojun Xu, Jinghan Jia, Yuanshun Yao, Yang Liu, and Hang Li. Robust multi-bit text watermark with llm-based paraphrasers. _arXiv preprint arXiv:2412.03123_ , 2024. 

- KiYoon Yoo, Wonhyuk Ahn, Jiho Jang, and Nojun Kwak. Robust multi-bit natural language watermarking through invariant features. _arXiv preprint arXiv:2305.01904_ , 2023. 

- KiYoon Yoo, Wonhyuk Ahn, and Nojun Kwak. Advancing beyond identification: Multi-bit watermark for large language models. In _Proceedings of the 2024 Conference of the North American Chapter of the Association_ 

19 

- _for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pages 4031–4055, 2024. 

- Jingqi Zhang, Ruibo Chen, Yingqing Yang, Peihua Mai, Heng Huang, and Yan Pang. Leave no trace: Blackbox detection of copyrighted dataset usage in large language models via watermarking. _arXiv preprint arXiv:2510.02962_ , 2025. 

- Ruisi Zhang, Shehzeen Samarah Hussain, Paarth Neekhara, and Farinaz Koushanfar. _{_ REMARK-LLM _}_ : A robust and efficient watermarking framework for generative large language models. In _33rd USENIX Security Symposium (USENIX Security 24)_ , pages 1813–1830, 2024. 

- Xuandong Zhao, Lei Li, and Yu-Xiang Wang. Permute-and-flip: An optimally robust and watermarkable decoder for llms. _arXiv preprint arXiv:2402.05864_ , 2024. 

20 

## **Appendix** 

## **A More Technical Details on the Methods** 

## **A.1 Hash Function Implementation** 

The PRF takes as input the candidate token _x_ , a context window **w** = ( _w_ 1 _, . . . , wk_ ) of _k_ token IDs, and the secret key _K_ (all of them are integers), and outputs a random integer in [0 _, M_ ). We compute the hash as follows: 

**==> picture [329 x 46] intentionally omitted <==**

where _q_ 1 _, . . . , qk_ are distinct large primes (to ensure that different orderings of the same tokens produce different values), and _p_ 2 _, p_ 3 _, p_ 4 are additional primes. The first result _h[′]_ undergoes XOR-shift for better bit dispersion: _h_ = ( _h[′] · p_ mix) _⊕_ (( _h[′] · p_ mix) _≫ s_ ), where _p_ mix is a mixing prime and _s_ is a shift constant. Finally, we normalize to obtain the uniform pseudo-random value: 

**==> picture [100 x 22] intentionally omitted <==**

## **A.2 Details on SynthID-Text Evaluation** 

In our experiments, we evaluate _SynthID-Text_ (Dathathri et al., 2024) as the state-of-the-art generationtime, distortion free and non deterministic watermark. While traditional methods (such as GumbelMax or Soft Red List) apply a single, global shift to the logit distribution, SynthID-Text embeds its signal through a multi-layered _Tournament sampling_ mechanism. 

_Tournament Generation._ At each generation step _t_ , the method seeds a pseudo-random function using the preceding _k_ tokens (the context window). Using this seed, the vocabulary _V_ is pseudorandomly partitioned into a tournament structure with _m_ layers. At each layer _l ∈{_ 1 _, . . . , m}_ , a pseudo-random _g_ -value _gt,l_ is computed. Instead of a single binary split, SynthID-Text iteratively reshapes the target LLM’s probability distribution across these _m_ layers. Tokens that consistently win their tournament matches (i.e., those assigned high _g_ -values across multiple layers) see their sampling probabilities exponentially increased. By distributing the watermark across multiple layers, SynthIDText preserves text quality while embedding a robust signal. In our implementation, we follow the authors’ specification for a binary random function (Bernoulli _g_ -value distribution) to construct this tournament. 

_Why we avoid SynthID’s Bayesian detector._ The original SynthID-Text framework proposes a Bayesian neural network (logistic regression or MLP) trained on a representative dataset to estimate posterior probabilities _P_ ( _w|g_ ). We avoid this approach for several reasons. 

_(i) No false-positive-rate guarantee._ A Bayesian posterior score has no frequentist calibration: there is no principled way to set a decision threshold that guarantees, for example, at most one false accusation in 10[4] documents. This is essential for any legal or regulatory use of watermark detection, where a false positive can constitute a wrongful accusation of AI generation. 

_(ii) Distribution dependence._ The trained classifier learns the joint distribution of token scores and watermark presence from its training corpus. Deploying on a different model, domain, language, or decoding strategy invalidates these learned posteriors; in practice, we observed that the Bayesian detector degrades sharply on out-of-domain text, requiring retraining for every new deployment setting. 

_(iii) Incompatibility with multiple-testing correction._ Localized detection (subsection 3.3) requires evaluating thousands of candidate windows and applying Bonferroni correction to control the family-wise 

21 

error rate. This demands a well-calibrated null distribution for each window, which a learned classifier cannot provide. The Bayesian scores are not _p_ -values and cannot be combined or corrected in a statistically valid manner. 

_(iv) Opacity and reproducibility._ A learned classifier is a black box whose decision boundary cannot be formally audited. For provenance claims that may carry legal weight, a closed-form statistical test with an analytically derived null distribution is far more defensible. Moreover, the Bayesian detector is not open-sourced, and despite following the specification in the original supplementary material, we were unable to reproduce comparable results, making fair comparison infeasible. 

_Our frequentist alternative._ To ensure a fair, threshold-independent comparison, we implement a mathematically rigorous frequentist detection pipeline. At detection time, for a given token _xt_ and its context, we reconstruct the PRF-seeded tournament and extract the sequence of _m_ layer-wise _g_ -values _gt,_ 1 _, . . . , gt,m_ . Because earlier layers in the tournament contribute more watermarking evidence than later layers, we compute a _Weighted Mean Score_ for the token: 

**==> picture [254 x 29] intentionally omitted <==**

where _α_ 1 _≥· · · ≥ αm ≥_ 0 are linearly decaying weights. Over a sequence of _N_ valid tokens, we sum the scores to obtain a test statistic _S_ =[�] _[N] t_ =1 _[s][t]_[.][Under][the][null][hypothesis] _[H]_[0][(unwatermarked][text),] the _g_ -values follow the unwatermarked uniform or Bernoulli distribution. We analytically compute the mean _µ_ 0 and variance _σ_ 0[2][of][the][weighted][sum][under] _[H]_[0][.][We][then][compute][a][final][Z-score][for][the] sequence: 

**==> picture [269 x 25] intentionally omitted <==**

The significance is given by the standard normal survival function _p_ = 1 _−_ Φ( _Z_ SynthID). 

## **A.3 Other Watermark Schemes** 

We describe below the other watermarking schemes referenced in this work. 

_Green-list/Red-list._ Kirchenbauer et al. (2023a) modify the logit vector based on the watermark context window and secret key _K_ . A token _v_ is classified as “green” if PRF( _v,_ **w** _, K_ ) _< γ_ (typically _γ_ =0 _._ 5), and its logit is incremented by _δ_ : _ℓ_[˜] _v_ = _ℓv_ + _δ_ for green tokens, _ℓ_[˜] _v_ = _ℓv_ otherwise. Detection counts green tokens and performs a binomial test. This method is _not_ distortion-free: the additive bias alters every generation. 

_MorphMark._ Wang et al. (2025) adaptively adjust watermark strength based on context. Let _PG_ = � _v∈_ GreenList _[p][v]_[be the total probability mass on green tokens.][If] _[ P][G][≤][p]_[0][(a threshold),][no watermark] is applied; otherwise, probabilities are rescaled with an adaptive boost factor _r_ = min( _κPG,_ 1). This reduces distortion compared to vanilla green-red but is still _not_ distortion-free. 

_DiPMark._ Wu et al. (2023) introduce a distortion-free variant of green-red watermarks using a pseudorandom permutation _π_ (seeded by context and _K_ ) to reorder tokens before applying a bias. The bias preserves the original distribution in expectation over the randomness of _π_ . _WaterMax._ Giboulot and Furon (2024) generate several candidate chunks from the original LLM distribution and select outputs with the highest watermark score. This is distortion-free by construction but requires multiple generations per query, making it impractical for production. 

## **A.4 Radioactivity Test Protocol** 

We detail the radioactivity test methodology from Sander et al. (2024, 2025). 

22 

_Teacher-forcing setup._ We feed the watermarked training traces into the suspect (student) model using teacher forcing: at each position _t_ , the model receives the ground-truth prefix _x<t_ from the ˆ watermarked trace and produces a prediction. Let _xt_ = arg max _v∈V Pθ_ ( _v | x<t_ ) denote the student’s top-1 prediction at step _t_ . The key insight is that teacher forcing isolates the model’s learned token preferences from confounding factors like sampling noise, requiring only a single forward pass over existing traces rather than expensive autoregressive generation. 

_Test statistic._ We score each prediction using the watermark’s PRF: _Rt_ = PRF(ˆ _xt,_ **w** _t, K_ ), where **w** _t_ is the context window of teacher tokens preceding position _t_ . If the student internalized the watermark bias during training, its top-1 predictions will be systematically skewed toward high-PRF tokens, producing a significant test statistic. 

_Deduplication._ We deduplicate at two levels, each for a different reason: (i) _within each trace_ , each context window **w** _t_ is scored only once. This is necessary because the teacher’s watermarked tokens appear in the student’s input context during teacher forcing: if an n-gram that was biased toward high-PRF values appears multiple times, the student might simply copy it from context rather than predicting it from internalized preferences, creating a false signal (Sander et al., 2024); (ii) _across traces_ , all (context window, predicted token) pairs are pooled and deduplicated, because the PRF is deterministic and shared (context, token) tuples across different training examples would yield identical scores, violating independence (Fernandez et al., 2023). After deduplication, under _H_ 0 (the student is unaware of _K_ ), the scores are independent and follow their null distribution, enabling exact _p_ -value computation. 

## **A.5 Formal Definitions of Non-Distortion** 

Following Dathathri et al. (2024), we provide the formal definitions of the two levels of non-distortion used throughout the paper. Let ∆ _V_ denote the probability simplex over the vocabulary _V_ , let _R_ be the space of random seeds, and let _K_ be the space of secret keys. 

**Definition 1** (Single-token non-distortion (Dathathri et al., 2024, Def. 16)) **.** A sampling algorithm _S_ : ∆ _V ×R →V_ is _single-token non-distortionary_ if for any probability distribution _**p** ∈_ ∆ _V_ and token _x ∈V_ : 

**==> picture [142 x 12] intentionally omitted <==**

Marginalizing over the random seed, the sampling algorithm produces each token with exactly its original LLM probability. This is a property of the sampling algorithm alone (e.g., Gumbel-max or two-sample Tournament sampling), independent of how the seed is generated across time steps. 

**Definition 2** (Single-sequence non-distortion (Dathathri et al., 2024, Def. 20 with _K_ =1)) **.** A watermarking scheme _P_ wm is _single-sequence non-distortionary_ if, for any prompt _**x**_ and response _**y** ∈V[∗]_ : 

**==> picture [174 x 12] intentionally omitted <==**

This is strictly stronger than single-token non-distortion: it requires the _joint_ distribution over a full response to match the original LLM, not just the per-token marginals. A single-token non-distortionary sampler paired with a sliding-window seed generator violates this property whenever the same context window repeats (producing deterministic rather than stochastic outputs). Repeated context masking (Dathathri et al., 2024; Kuditipudi et al., 2023) restores single-sequence non-distortion by falling back to unwatermarked sampling on repeated contexts (Remark 1). 

## **B Gumbel-max proofs** 

The following results were presented by Aaronson and Kirchner (2023) and formalized by Fernandez et al. (2023). Some elements of these proofs are used later, so we restate them here. An overview of the Gumbel-max generation scheme is presented in Figure 10. 

23 

**Proposition 4** (Sampling probability, restated from Proposition 1) **.** _Consider a discrete distribution iid_ _**p**_ = ( _p_ 1 _, . . . , pV_ ) _and V_ = _|V| random variables_ _**R**_ = ( _R_ 1 _, . . . , RV_ ) _s.t. Rv ∼U_ [0 _,_ 1] _. Let V[⋆]_ = arg max _v Rv_[1] _[/p][v] . Then:_ P( _V[⋆]_ = _v_ ) = _pv._ 

_iid Proof of Proposition 1._ For any _v ∈V_ , _Rv ∼U_ [0 _,_ 1] so, _−_ ln( _Rv_ ) follows an exponential distribution _E_ (1). Let _Zv_ := _− p_[1] _v_[ln(] _[R][v]_[)][.][By][construction,] _[Z][v][∼E]_[(] _[p][v]_[)][,][with][density] _[f][Z][v]_[(] _[z]_[)][=] _[p][v][e][−][p][v][.z]_[.][We][now] have: 

**==> picture [294 x 20] intentionally omitted <==**

A well known result about exponential laws is that: 

**==> picture [289 x 31] intentionally omitted <==**

**==> picture [326 x 23] intentionally omitted <==**

This shows that for a given secret vector _**r**_ , the watermarking chooses a word which may be unlikely (low probability _pV[⋆]_ ). Yet, on expectation over the secret keys, i.e., over r.v. _**R**_ = ( _R_ 1 _, . . . , RV_ ), the distribution of the chosen token follows the distribution given by the LLM. 

**Corollary 2** (Restated from Corollary 1) **.** _Conditionally on V[⋆]_ = _v, RV ⋆ ∼ Beta_ (1 _/pv,_ 1) _._ 

_Proof of Corollary 1._ From the proof above, _Z_ = min _v Zv ∼E_ (1) and _V[⋆]_ = arg min _v Zv_ . A standard property of competing exponentials is that the identity of the winner is independent of the winning time: _V[⋆] ⊥ Z_ . Conditioning on _V[⋆]_ = _v_ , we therefore still have _Z ∼E_ (1), and: 

**==> picture [289 x 23] intentionally omitted <==**

**==> picture [426 x 251] intentionally omitted <==**

**----- Start of picture text -----**<br>
w = ( xt−k, . . . , xt− 1) w<br>Generated: x 1 x 2 · · · xt−k · · · xt− 1 Candidates v ∈V :<br>LLM context (all tokens) v 1 v 2 · · · vV<br>each v<br>LLM PRF( w , K )<br>p [(] [t] [)] = ( p 1 , . . . , pV  ) R  = ( R 1 , . . . , RV  )<br>Rv = PRF( v,  w , K )<br>xt = arg max v Rv [1] [/p][v]<br>xt<br>max<br>Selected token v 2<br>Rv [1] [/p][v] : has highest Rv [1] [/pv]<br>v 1 v 2 v 3 v 4 v 5<br>**----- End of picture text -----**<br>


**Figure 10** Standard Gumbel-Max watermarking (see section 2). The LLM uses all previous tokens to predict probabilities, while the PRF uses only the last _k_ tokens (watermark context **w** ) to generate pseudo-random values _Rv_ for each candidate token _v_ . The token maximizing _Rv_[1] _[/p][v]_ is selected. 

24 

which gives _Rv_ = _e[−][p][v][E]_ with _E ∼E_ (1), with p.d.f. _fRv_ ( _r_ ) = _[r]_[1] _[/pv] pv[−]_[1] . Therefore, _Rv | V[⋆]_ = _v ∼_ Beta(1 _/pv,_ 1). 

**Proposition 5** (Expected score under _H_ 1, restated from Proposition 3) **.** _Under H_ 1 _(text is water-_ 2 _π marked),_ E( _ST_ ) _≥ T_ + � 6 _[−]_[1] � _HT , where HT_ = _−_[�] _[T] t_ =1 _[p][t]_[ ln(] _[p][t]_[)] _[is][the][entropy][of][the][completion.]_ 

_Proof of Proposition 3._ From the corollary above, _Rt_ = exp( _−ptE_ ) with _E ∼E_ (1), so: 

**==> picture [184 x 100] intentionally omitted <==**

(by change of variable _x_ = _−_ 1 _/pt_ ln( _r_ ) ) 

Then, using integration by parts with _u_ = 1 _− r_[1] _[/p][t]_ and _v_ = ln(1 _− r_ ), the integral becomes: 

**==> picture [235 x 26] intentionally omitted <==**

where _Hz_ is the _z_ -th harmonic number also defined as _Hz_ =[�] _[∞] n_ =1 _n_ 1 _[−] n_ +1 _z_[.][Therefore,][we][have:] 

**==> picture [244 x 61] intentionally omitted <==**

Now, _∀n ∈_ N _[⋆]_ , we have: 

**==> picture [267 x 93] intentionally omitted <==**

Therefore, by summing over all _t ∈_ [1 _, T_ ], 

**==> picture [200 x 59] intentionally omitted <==**

## **C Proofs on Diversity Schemes for Gumbel Max** 

We derive bounds on the expected detection score E[ _ST_ ] under _H_ 1 for each diversity strategy described in subsection 5.1 and illustrated in Figure 11. All bounds decompose as the standard Gumbel bound (Proposition 3) plus a correction term capturing the cost of the diversity mechanism. 

25 

Recall that under the standard Gumbel scheme, _Rt ∼_ Beta(1 _/pt,_ 1) and the expected per-token score is E[ _st_ ] = _H_ 1 _/pt_ , leading to E[ _ST_ ] _≥ T_ + ( _[π]_ 6[2] _[−]_[1)] _[H][T]_[ .][In][each][case][below,][some][tokens][are][either] unwatermarked or have a modified distribution of _Rt_ . For unwatermarked tokens, _Rt ∼U_ [0 _,_ 1] and E[ _st_ ] = 1. 

## **C.1 Dual-Key Routing** 

Dual-key routing (subsection 3.1) maintains two secret keys _k_[(1)] and _k_[(2)] . At each generation step, key _k_[(1)] is selected with probability 1 _− α_ and _k_[(2)] with probability _α_ . The token is produced via GumbelMax using the selected key. Detection aggregates scores from both keys: _si_ = (1 _− α_ ) _· s_[(1)] _i_ + _α · s_[(2)] _i_[.] 

**Proposition 6** (Bound on expected score under dual-key routing, single-key detection) **.** _Under dualkey routing with parameter α ∈_ [0 _,_ 1] _(key k_[(1)] _selected with probability_ 1 _− α, key k_[(2)] _with probability α), detection using a single key k_[(1)] _yields:_ 

**==> picture [302 x 26] intentionally omitted <==**

_Proof._ At each step _t_ , key _k_[(1)] is selected with probability 1 _− α_ and key _k_[(2)] with probability _α_ . For detection using key _k_[(1)] : 

- With probability 1 _−α_ : the PRF value _Rt_[(1)] is the one used for generation, so _Rt_[(1)] _∼_ Beta(1 _/pt,_ 1) and E[ _s_[(1)] _t_[] =] _[ H]_ 1 _/pt_[.] 

- With probability _α_ : the token was generated using key _k_[(2)] , so _Rt_[(1)] is independent of the generation process. It is effectively uniform and E[ _s_[(1)] _t_[] = 1][.] 

Summing over _T_ tokens: 

**==> picture [142 x 29] intentionally omitted <==**

**==> picture [380 x 50] intentionally omitted <==**

This bound matches the random skip bound (Proposition 10) with _α_ playing the role of the skip rate: from the perspective of a single-key detector, tokens generated with the other key look exactly like skipped tokens. The advantage of dual-key routing is that the aggregated score (Equation 3) lets every token contribute signal from at least one key, as formalized below. 

**==> picture [442 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
Standard Stochastic Entropy Random Adaptive Ent-Norm Dual-Key<br>Gumbel Mixing Warmup Skip Skip Skip Routing<br>PRF( w , K ) PRF r 0 � Hi > τs ? Coin flip α Gumbel Gumbel Route α<br>no yes skip keep 1 −α α<br>R Mix r 1 , r 0 Sample p Gumbel Sample p Gumbel RV ⋆< τ ? RV ⋆< τ [pV ⋆] ? PRF( k [(1)] ) PRF( k [(2)] )<br>yes no yes no<br>arg max  Rv [1] [/pv] arg max  rv [1] [/pv] Sample p Keep V  [⋆] Sample p Keep V  [⋆] arg max  Rv [1] [/pv]<br>xt xt xt xt xt xt xt<br>Deterministic Stochastic Stochastic (prefix) Stochastic Stochastic Stochastic Stochastic<br>✓ Distortion-free ✓ Distortion-free ✓ Distortion-free ✓ Distortion-free × Not distortion-free ✓ Distortion-free ✓ Distortion-free<br>**----- End of picture text -----**<br>


**Figure 11** Overview of diversity mechanisms for Gumbel-Max watermarking. Each column shows how the token _xt_ is generated. All methods except Adaptive Skip preserve the distortion-free property. The key distinction lies in _where_ randomness is injected: in the PRF value (Mixing), in the decision to watermark (Skip variants, Warmup), or in the key selection (Dual-Key Routing). 

26 

**Proposition 7** (Expected score under dual-key Early Fusion detection) **.** _Under dual-key routing with parameter α ∈_ [0 _,_ 1] _, detecting with the aggregated score si_ = (1 _− α_ ) _· s_[(1)] _i_ + _α · s_[(2)] _i yields:_ 

**==> picture [317 x 25] intentionally omitted <==**

_Proof._ At each step _t_ , key _k_[(1)] is active with probability 1 _− α_ and key _k_[(2)] with probability _α_ . The aggregated per-token score is _Tt_ = (1 _− α_ ) _s_[(1)] _t_ + _αs_[(2)] _t_[.] 

- If _k_[(1)] was used (prob. 1 _− α_ ): _s_[(1)] _t_ has the watermarked distribution (E[ _s_[(1)] _t_[]][=] _[H]_ 1 _/pt_[)][and] _[s]_[(2)] _t_ is uniform (E[ _s_[(2)] _t_[] = 1][),][giving][E][[] _[T][t]_[] = (1] _[ −][α]_[)] _[H]_ 1 _/pt_[+] _[ α]_[.] 

**==> picture [428 x 28] intentionally omitted <==**

Taking expectation over the key choice: 

**==> picture [250 x 44] intentionally omitted <==**

where _θR_ = _α_[2] + (1 _− α_ )[2] . Summing over _T_ tokens: 

**==> picture [148 x 30] intentionally omitted <==**

**==> picture [374 x 50] intentionally omitted <==**

Note that _θR_ = _α_[2] + (1 _− α_ )[2] _≤_ 1 _− α_ for _α ≤_ 0 _._ 5, so the expected score under Early Fusion is actually lower than under single-key detection (Proposition 6). The power advantage of Early Fusion comes not from a higher expected score, but from the reduced null variance ( _θR_ per token instead of 1), which yields a better Z-score as shown below. 

## **C.1.1 Power Analysis: Early vs. Late Fusion** 

We analyze the statistical power of the Early Fusion test compared to a classical single-key baseline and alternative Late Fusion strategies using the Z-score (Signal-to-Noise Ratio) separation: 

**==> picture [105 x 27] intentionally omitted <==**

Assume a standard Gumbel-Max test where an unwatermarked token yields an expected score of 1 with a variance of 1, and a successfully watermarked token yields an expected score _µw >_ 1. 

_Single-Key Baseline._ For a traditional single-key test with _n_ tokens, the expected score sum under _H_ 1 is _nµw_ , and under _H_ 0 is _n_ . The null variance is _n_ . 

**==> picture [140 x 22] intentionally omitted <==**

27 

_Early Fusion: Unweighted (w_ = 0 _._ 5 _)._ For the unweighted test, the expected score per token is E[¯ _si_ ] = _[µ][w]_ 2[+][1] regardless of which key generated it. The null variance is Var(¯ _si_ ) =[1][2] 2[+1][2][2] = 0 _._ 5. 

**==> picture [221 x 27] intentionally omitted <==**

Thus, _Z_ early = ~~_√_~~ 12 _[Z]_[base] _[≈]_[0] _[.]_[707] _[Z]_[base][.][This][proves][that][unweighted][Early][Fusion][is][perfectly] **[invariant] to** _α_ , but requires exactly twice as many tokens (2 _n_ ) as the single-key baseline to reach the same statistical confidence. 

_Early Fusion: Optimal Weighted (w_ = _α)._ If the routing probability _α_ is known (e.g., via speculative decoding acceptance rates) and we use optimal weights _w_ = _α_ , the expected token score under _H_ 1 becomes E[ _si_ ] = _α_ ( _αµw_ + 1 _− α_ ) + (1 _− α_ )((1 _− α_ ) _µw_ + _α_ ). Simplifying this and calculating the Z-score yields: 

_Zα_ = _[√] n_ ( _µw −_ 1)� _α_[2] + (1 _− α_ )[2] When _α_ = 0 _._ 5 (maximum diversity), _Zα_ = _Z_ early _≈_ 0 _._ 707 _Z_ base. When _α_ = 0 _._ 1 (typical for draft model acceptance in speculative decoding), _Zα_ = _√_ 0 _._ 1[2] + 0 _._ 9[2] _Z_ base _≈_ 0 _._ 905 _Z_ base. This demonstrates that the weighted test recovers nearly 30% of the statistical power lost to diversity when the generation rate is skewed. 

_Superiority over Late Fusion._ We can now formally demonstrate why token-level aggregation outperforms independent per-key testing (late fusion). Late fusion evaluates each key’s scores independently across the entire sequence ( _S_[(1)] =[�] _s_[(1)] _i_ and _S_[(2)] =[�] _s_[(2)] _i_[) and then combines their resulting p-values] (e.g., via Fisher’s method or by taking the minimum p-value). 

Assuming without loss of generality that _α ≥_ 0 _._ 5, the expected signal for the dominant key over the null is _nα_ ( _µw −_ 1). The variance remains _n_ . The statistical power of the combined Late Fusion test is ultimately bounded by the strongest independent signal it receives, which achieves at best: 

**==> picture [200 x 24] intentionally omitted <==**

To prove optimal Early Fusion natively dominates Late Fusion, we compare their Z-scores. We must show that _Zα > Z_ late, which simplifies to proving � _α_[2] + (1 _− α_ )[2] _> α_ for any _α ∈_ (0 _,_ 1): 

**==> picture [226 x 13] intentionally omitted <==**

We test the inequality 2 _α_[2] _−_ 2 _α_ + 1 _> α_[2] : 

**==> picture [152 x 12] intentionally omitted <==**

Since ( _α −_ 1)[2] is strictly positive for all _α ∈_ (0 _,_ 1), it follows that _Zα > Z_ late. Therefore, token-level aggregation strictly dominates independent per-key testing by preserving the complementary signal distributed across both keys ( _k_[(1)] and _k_[(2)] ) at the token level, rather than systematically treating the minority key’s tokens as noise during independent sequence-level evaluations. 

## **C.2 Stochastic Mixing** 

Stochastic mixing introduces true randomness by mixing the deterministic PRF value _r_ 1 with a Bernoulli coin. Given a parameter _a ∈_ (0 _,_ 1), the mixed value is _r_ = _a · r_ 1 with probability _a_ , or _r_ = _a_ +(1 _− a_ ) _· r_ 1 with probability 1 _− a_ . The mixed _r_ remains uniform (distortion-free), but detection uses only _r_ 1. 

**Proposition 8** (Bound on expected score under mixing) **.** _Under stochastic mixing with parameter a ∈_ (0 _,_ 1) _, detection is performed using r_ 1 _(the deterministic PRF value). The expected score satisfies:_ 

**==> picture [339 x 30] intentionally omitted <==**

28 

_Proof._ Let _R ∼_ Beta(1 _/p,_ 1) be the random variable selected during sampling. The score for a single token is _s_ = _−_ ln(1 _−r_ 1), where _r_ 1 is recovered from _R_ as: _r_ 1 = _R/a_ if _R ∈_ [0 _, a_ ], and _r_ 1 = ( _R−a_ ) _/_ (1 _−a_ ) if _R ∈_ [ _a,_ 1]. 

We decompose E[ _s_ ] by interval: 

**==> picture [276 x 40] intentionally omitted <==**

**==> picture [411 x 46] intentionally omitted <==**

since P( _R > a_ ) = 1 _− a_[1] _[/p]_ . 

_a_ For _I_ 1: since _r/a ≥ r_ for _r ∈_ [0 _, a_ ], we have _−_ ln(1 _−r/a_ ) _≥−_ ln(1 _−r_ ), giving _I_ 1 _≥_ �0 _[−]_[ln(1] _[−][r]_[)] _[f][R]_[(] _[r]_[)] _[dr]_[.] Summing yields E[ _s_ ] _≥_ E[ _s_ std] + (1 _− a_[1] _[/p]_ ) ln(1 _− a_ ) where E[ _s_ std] = _H_ 1 _/p_ . Applying the standard bound and summing over _T_ tokens gives the result. 

**Proposition 9** (Distortion-freeness of mixing) **.** _The mixed variable r follows U_ [0 _,_ 1] _, so the sampled token follows the model distribution_ _**p** ._ 

_Proof._ Let _FR_ ( _x_ ) = P( _r ≤ x_ ). For _x ≤ a_ : _r ≤ x_ requires _r_ 0 = 0, giving P( _r ≤ x_ ) = _a ·_ P( _r_ 1 _≤ x/a_ ) = _a · x/a_ = _x_ . For _x > a_ : P( _r ≤ x_ ) = _a_ + (1 _− a_ ) _·[x]_ 1 _−[−] a[a]_[=] _[ x]_[.][Since] _[F][R]_[(] _[x]_[) =] _[ x]_[,][we][have] _[r][∼U]_[[0] _[,]_[ 1]][.] 

_Behavior of the penalty._ The penalty (1 _− a_[1] _[/p]_ ) ln(1 _− a_ ) is always non-positive (since ln(1 _− a_ ) _<_ 0) and vanishes at both extremes: as _a →_ 0, ln(1 _− a_ ) _→_ 0; as _a →_ 1, (1 _−_ (1 _− ϵ_ )[1] _[/p]_ ) ln( _ϵ_ ) _∼ p[ϵ]_[ln(] _[ϵ]_[)] _[ →]_[0][.] This is expected since in these extremes, all tokens take the same route, which makes it similar to vanilla Gumbel-max. 

## **C.3 Random Skip** 

Random skip disables the watermark independently at each token with probability _α_ , reverting to standard sampling from _**p**_ . This blindly injects randomness to break deterministic loops, uniformly attenuating the detection signal. 

**Proposition 10** (Bound on expected score under periodic skip) **.** _Under periodic skip with rate α ∈_ [0 _,_ 1] _(each token is independently skipped with probability α), the expected score satisfies:_ 

**==> picture [299 x 25] intentionally omitted <==**

_Proof._ At each step _t_ , with probability 1 _−α_ the watermark is active and E[ _st_ ] = _H_ 1 _/pt_ ; with probability _α_ the watermark is skipped and E[ _st_ ] = 1. Summing: 

**==> picture [137 x 30] intentionally omitted <==**

**==> picture [378 x 47] intentionally omitted <==**

The entropy-dependent signal is uniformly attenuated by a factor (1 _− α_ ), regardless of the token entropy. This is wasteful compared to adaptive strategies that selectively skip only low-signal tokens. 

29 

## **C.4 Adaptive Skip** 

Adaptive skip disables the watermark selectively when the model is highly confident. At each step, the token is produced via Gumbel-Max, but if the winning PRF value _RV ⋆_ falls below a threshold _τ_ , the watermark is discarded and the token is resampled from _**p**_ . Low _RV ⋆_ indicates the token won due to high probability mass rather than a favorable PRF draw, so skipping it sacrifices little detection signal. 

**Proposition 11** (Adaptive skip is not distortion-free) **.** _Under adaptive skip with threshold τ ∈_ (0 _,_ 1) _, the output distribution is:_ 

**==> picture [327 x 31] intentionally omitted <==**

_which differs from pv unless_ _**p** is uniform._ 

_Proof._ Let _V[⋆]_ be the initial token selected by the Gumbel-max trick, where P( _V[⋆]_ = _v_ ) = _pv_ . By Corollary 1, the conditional distribution of the pseudo-random value _Rv_ is Beta(1 _/pv,_ 1). The watermark is skipped if _RV[⋆] < τ_ . 

The marginal probability of outputting a specific token _v_ decomposes into two disjoint events: keeping the initially selected _v_ , or skipping and resampling _Xt_ = _v_ from the original distribution _**p**_ : 

**==> picture [432 x 145] intentionally omitted <==**

For the mechanism to be distortion-free, we require P(output = _v_ ) = _pv_ for all _v ∈V_ . This implies: 

**==> picture [89 x 22] intentionally omitted <==**

The right-hand side is a constant across all tokens, whereas the left-hand side strictly depends on _pv_ . This equality holds if and only if all tokens have the exact same probability _pv_ = 1 _/|V|_ . 

**Remark 2.** The distortion shifts mass from high-confidence tokens (large _pv_ , frequently skipped since _τ_[1] _[/p][v]_ is large) toward low-confidence tokens (small _pv_ , rarely skipped). For example, with _p_ 1 = 0 _._ 9, _p_ 2 = 0 _._ 1, and _τ_ = 0 _._ 5: the output probabilities become (0 _._ 858 _,_ 0 _._ 142) instead of (0 _._ 9 _,_ 0 _._ 1). In practice, _τ_ is small (e.g., _τ_ = 0 _._ 1), so the distortion is mild. 

**Proposition 12** (Bound on expected score under adaptive skip) **.** _Under adaptive skip with threshold τ ∈_ [0 _,_ 1] _(the watermark is disabled when RV_[(] _[t]_[)] _[⋆][< τ][),][the][expected][score][satisfies:]_ 

**==> picture [328 x 30] intentionally omitted <==**

_The correction term is always non-positive, vanishing as τ →_ 0 _._ 

_Proof._ We condition on the identity of the selected token. By Proposition 1, P( _V[⋆]_ = _v_ ) = _pv_ . By Corollary 1, _conditioned on V[⋆]_ = _v_ , the PRF value _Rv ∼_ Beta(1 _/pv,_ 1) with density _f_ ( _r_ ) = _p_ 1 _v[r]_[1] _[/p][v][−]_[1] 

30 

and CDF _F_ ( _r_ ) = _r_[1] _[/p][v]_ . The skip condition _Rv < τ_ therefore has conditional probability P( _Rv < τ | V[⋆]_ = _v_ ) = _τ_[1] _[/p][v]_ . This decreases with entropy: for confident tokens ( _pv →_ 1), _τ_[1] _[/p][v] → τ_ (frequent skipping); for unlikely tokens ( _pv →_ 0), _τ_[1] _[/p][v] →_ 0 (rare skipping). 

We now bound E[ _st | V[⋆]_ = _v_ ]. Decomposing over the skip decision: 

**==> picture [372 x 26] intentionally omitted <==**

The first term integrates the score over the non-skip region using the conditional density of _Rv_ : 

**==> picture [262 x 26] intentionally omitted <==**

For the second term, the replacement token _Xt ∼_ _**p**_ is drawn with independent randomness, but its PRF value _RXt_ comes from the _same_ realization _**R**_ , so we cannot claim its expected score is 1 (see Remark 4). Since _−_ ln(1 _− RXt_ ) _≥_ 0, the second term is non-negative, so: 

E[ _st | V[⋆]_ = _v_ ] _≥_ 1 _−_ ln(1 _− r_ ) _r_[1] _[/p][v][−]_[1] _dr_ = _H_ 1 _/pv − τ −_ ln(1 _− r_ ) _r_[1] _[/p][v][−]_[1] _dr_ � _τ pv_ �0 _pv_ where we used � _τ_ 1[=] �01 _[−]_ �0 _τ_[and] �01 _−_ ln _p_ ( _v_ 1 _−r_ ) _r_[1] _[/p][v][−]_[1] _dr_ = _H_ 1 _/pv_ . Since _−_ ln(1 _− r_ ) _≤−_ ln(1 _− τ_ ) for _r ∈_ [0 _, τ_ ]: 

**==> picture [244 x 122] intentionally omitted <==**

and therefore E[ _st | V[⋆]_ = _v_ ] _≥H_ 1 _/pv_ + _τ_[1] _[/p][v]_ ln(1 _− τ_ ). Since this holds for every _v_ , it holds for the realized token probability _pt_ = _pV ⋆_ . Summing over _T_ steps and applying the standard bound (Proposition 3) to[�] _t[H]_[1] _[/p] t[≥][T]_[+ (] _[π]_ 6[2] _[−]_[1)] _[H][T]_[gives][the][result.] 

**Remark 3.** The penalty ln(1 _− τ_ )[�] _t[τ]_[ 1] _[/p][t]_[is][always][non-positive][(since][ln(1] _[ −][τ]_[)] _[<]_[0][),][confirming] that skipping can only reduce the detection signal. For small _τ_ (e.g., _τ_ = 0 _._ 1), the penalty is negligible: _τ_[1] _[/p][t]_ is small for all but deterministic tokens ( _pt ≈_ 1), and those tokens carry no watermark signal anyway ( _H_ 1 = 1, equal to the null baseline). The bound is conservative because we dropped the skip contribution entirely; in practice, skipped tokens still contribute positively to the score. 

**Remark 4** (Skip contribution) **.** A tempting (but incorrect) approach is to claim that skipped tokens contribute expected score 1, arguing that the replacement token _Xt ∼_ _**p**_ is drawn independently and therefore its PRF value _RXt_ is uniform. This would yield the decomposition: 

**==> picture [188 x 26] intentionally omitted <==**

leading to a correction _τ_[1] _[/p][t]_ (1 + ln(1 _− τ_ )) that is _positive_ for _τ <_ 1 _−_ 1 _/e_ —implying that skipping _improves_ detection, which is impossible. 

The error is that while _Xt_ is drawn independently of _**R**_ , the PRF value _RXt_ = _**R**_ [ _Xt_ ] shares the _same_ realization _**R**_ . Since the skip event _{RV ⋆ < τ }_ constrains _**R**_ (the winning PRF value is low), the conditional expectation E[ _−_ ln(1 _− RXt_ ) _| RV ⋆ < τ_ ] _̸_ = 1. A simple counterexample: for a deterministic token ( _pt_ = 1), there is only one possible token, so skipping changes nothing and E[ _st_ ] = _H_ 1 = 1. Yet the incorrect formula gives 1 + _τ_ (1 + ln(1 _− τ_ )) _>_ 1. 

31 

## **C.5 Entropy-Normalized Adaptive Skip** 

This variant of adaptive skip replaces the fixed threshold _τ_ with an entropy-dependent threshold _τ[p][V ⋆]_ , which ensures every token is skipped with exactly the same probability _τ_ regardless of its confidence level. This restores the distortion-free property lost by standard adaptive skip. 

For a target skip rate _τ ∈_ (0 _,_ 1), the watermark is now disabled and the token is resampled from _**p**_ if: 

**==> picture [52 x 10] intentionally omitted <==**

**Proposition 13** (Distortion-freeness of entropy-normalized skip) **.** _The entropy-normalized adaptive skip mechanism is distortion-free, i.e., :_ P( _output_ = _v_ ) = _pv for all v ∈V._ 

_Proof._ We first evaluate the conditional probability of a skip occurring given that token _v_ was initially selected. By Corollary 1, _Rv | V[⋆]_ = _v ∼_ Beta(1 _/pv,_ 1), which has the cumulative distribution function _F_ ( _r_ ) = _r_[1] _[/p][v]_ . Therefore, the conditional skip probability is: 

**==> picture [248 x 14] intentionally omitted <==**

Because this conditional probability is exactly _τ_ for _every_ token in the vocabulary, the unconditional probability of a skip is also exactly _τ_ . Indeed, by the law of total probability: 

**==> picture [309 x 23] intentionally omitted <==**

The total marginal probability of outputting token _v_ can then be found by partitioning over the two mutually exclusive generation paths (whether a skip occurs or not): 

**==> picture [316 x 41] intentionally omitted <==**

We know the unconditional probability of a skip is _τ_ , so the conditional probability of not skipping is 1 _− τ_ . Furthermore, the replacement token _Xt_ is sampled from the original distribution independently of the skip event, so P( _Xt_ = _v_ ) = _pv_ . Substituting these values yields: 

**==> picture [157 x 40] intentionally omitted <==**

Thus, the marginal distribution is perfectly preserved, making the entropy-normalized mechanism distortion-free. 

**Proposition 14** (Bound on expected score under entropy-normalized skip) **.** _Under the entropynormalized adaptive skip with target skip rate τ ∈_ (0 _,_ 1) _, the expected score satisfies:_ 

**==> picture [324 x 30] intentionally omitted <==**

_Proof._ We condition on the identity of the selected token _V[⋆]_ = _v_ . We decompose the expected score into the non-skipped and skipped cases: 

**==> picture [390 x 12] intentionally omitted <==**

As discussed in Remark 4, the replacement token _Xt_ relies on the same PRF realization _**R**_ , so its contribution is difficult to isolate but strictly non-negative. Dropping the second term provides a conservative lower bound: 

**==> picture [354 x 27] intentionally omitted <==**

32 

Since the function _−_ ln(1 _− r_ ) is monotonically increasing, for _r ∈_ [0 _, τ[p][v]_ ], we have _−_ ln(1 _− r_ ) _≤ −_ ln(1 _− τ[p][v]_ ). We can bound the subtracted integral: 

**==> picture [268 x 83] intentionally omitted <==**

Substituting this back yields: 

**==> picture [166 x 13] intentionally omitted <==**

Since this inequality holds for any chosen _v_ , it holds for the realized token probability _pt_ . Summing over the sequence of _T_ tokens and applying the standard Gumbel bound (Proposition 3) gives: 

**==> picture [336 x 30] intentionally omitted <==**

The correction term is strictly non-positive because _τ[p][t] ∈_ (0 _,_ 1), meaning ln(1 _− τ[p][t]_ ) _<_ 0. This accurately reflects the expected loss in signal when skipping exactly _τ_ fraction of the tokens. 

**Remark 5** (Skip behavior and score penalty) **.** Unlike standard adaptive skip where the skip probability _τ_[1] _[/p][v]_ depends on token confidence (skipping high-confidence tokens more often), the entropynormalized threshold _τ[p][v]_ ensures a _uniform_ skip rate of exactly _τ_ for all tokens regardless of their probability. However, the per-token score penalty _τ_ ln(1 _− τ[p][t]_ ) still varies with entropy: 

- For high-confidence tokens ( _pt →_ 1): the penalty approaches _τ_ ln(1 _− τ_ ), which is mild. These tokens contribute little watermark signal anyway ( _H_ 1 = 1, equal to the null baseline), so skipping them has minimal impact. 

- For low-confidence tokens ( _pt →_ 0): the threshold _τ[p][t] →_ 1, making the penalty bound _τ_ ln(1 _− τ[p][t]_ ) _→−∞_ , making the bound effectively useless. Such tokens occur rarely (P( _V[⋆]_ = _v_ ) = _pv_ ), so their contribution to the total penalty is attenuated by their low occurrence frequency. 

The mechanism thus achieves distortion-freeness while concentrating the detection penalty on tokens that either carry little signal (high-confidence) or appear rarely (low-confidence), preserving most of the watermark power from medium-entropy tokens. 

## **D Fast Localization and Statistical Penalties** 

In practical settings, the exact start and end indices of a watermarked insertion are unknown. Our objective is to determine a set of disjoint watermarked intervals _{_ [ _a_ 1 _, b_ 1] _, . . . ,_ [ _ay, by_ ] _}_ . A naive exhaustive search over all possible intervals in a sequence of length _n_ requires evaluating � _n_ 2� _≈ n_[2] _/_ 2 windows. This _O_ ( _n_[2] ) search space not only introduces severe computational bottlenecks but also imposes an insurmountable statistical penalty via multiple-testing correction, as false positives become significantly more likely as the number of tested hypotheses grows. To optimize both computational and statistical efficiency, we utilize a geometric cover search space (Kirchenbauer et al., 2023b) combined with a fast two-stage extraction pipeline and rigorous Bonferroni correction. 

## **D.1 The Geometric Cover Search** 

To avoid testing _O_ ( _n_[2] ) intervals, we constrain our search to a dyadic grid of windows. We define a set of window lengths _L ∈{L_ 0 _,_ 2 _L_ 0 _,_ 4 _L_ 0 _, . . . ,_ 2 _[⌊]_[log][2] _[ n][⌋] }_ , where _L_ 0 = 2 _[⌈]_[log][2] _[ L]_[min] _[⌉]_ is the smallest power of two at least as large as the minimum zone length _L_ min. For each length _L_ , we slide the window across the text with a stride of _L/_ 2. 

33 

This geometric grid guarantees that any arbitrary watermarked region of length _L[∗] ≥ L_ min will be at least 50% covered by at least one grid window. The total number of candidate windows _M_ in this grid is strictly bounded: 

**==> picture [305 x 34] intentionally omitted <==**

By restricting the search to _M ≈O_ ( _n/L_ min) windows, we reduce the hypothesis space by orders of magnitude, dramatically lowering the statistical tax required to claim significance. 

## **D.2 Fast Two-Stage Pipeline and Greedy Extraction** 

Evaluating the rigorous, entropy-weighted Gamma distribution for all _M_ windows can be computationally heavy. To process arbitrarily large documents efficiently, we utilize a two-stage pipeline: 

1. **Fast Filtering:** We pre-calculate prefix sums of the unweighted raw scores _si_ . The sum for any candidate interval in the grid can then be computed in _O_ (1) time. We select the top candidates based on these raw sums. 

2. **Rigorous Scoring:** For the most promising candidates, we compute the exact entropy-weighted moment-matched Gamma _p_ -value _p_ raw (as defined in Section 3.2). 

To extract multiple zones, we proceed greedily. We find the window _I[∗]_ with the most significant _p_ raw. If its penalized significance (accounting for the search tax) is high enough, we flag it as watermarked, mask its tokens (setting their scores to zero), and repeat the search on the residual text. We aggregate disjoint intervals until the combined _p_ -value fails to overcome the multiple-testing threshold, up to a maximum of _Y_ max zones. 

## **D.3 The Bonferroni Tax and False Positive Guarantee** 

Evaluating _M_ intervals introduces the multiple comparisons problem. To maintain a strict family-wise error rate (FWER) _ϵ_ under the null hypothesis _H_ 0, we apply a union bound. 

For a single-zone search, the Bonferroni correction factor is simply _M_ . For a multi-zone search identifying _y_ disjoint regions, we must account for the number of ways to choose _y_ windows from the grid, _M_ � _y_ �, as well as the optimization over _y ∈_ [1 _, Y_ max]. The corrected _p_ -value in log-space is: 

**==> picture [313 x 25] intentionally omitted <==**

Under _H_ 0, the probability that the most significant window combination exceeds our threshold is strictly bounded: 

**==> picture [358 x 32] intentionally omitted <==**

where _K_ represents the total number of tested hypotheses in the search space. This guarantees that the probability of falsely accusing an entirely human-written text remains _≤ ϵ_ , regardless of document length. 

## **D.4 Asymptotic Power Comparison: Global vs. Localized Detection** 

We define the crossover point where the localized multi-zone test yields a stronger rejection of _H_ 0 than the global test. Let _n_ be the document length, and _ρ ∈_ (0 _,_ 1] be the fraction of tokens that are watermarked ( _w_ = _ρn_ ). 

˜ _Setup and Approximation._ Let the weighted token score _si_ have mean _µ_ 0 and variance _σ_[2] under _H_ 0. Under _H_ 1, the mean shifts to _µw > µ_ 0. Let _δ_ = ( _µw − µ_ 0) _/σ_ be the per-token signal-to-noise ratio. Using a Gaussian tail approximation, the log _p_ -value of a Z-score is ln _p ≈−_[1] 2 _[Z]_[2][.][We define][ ∆][2][=] _[ δ]_[2] _[/]_[2] as the expected log _p_ -value accumulation rate per watermarked token. 

34 

_Power of the Global Test._ The global test evaluates all _n_ tokens. The expected Z-score is: 

**==> picture [338 x 24] intentionally omitted <==**

The signal strength scales quadratically with _ρ_ ; the (1 _− ρ_ ) _n_ human tokens contribute no signal but inflate the variance, diluting the test. 

_Power of the Localized Test._ Assuming a localized test correctly isolates the _ρn_ watermarked tokens into _y_ zones, the variance is reduced to the watermarked subset _ρnσ_[2] . The expected raw log _p_ -value is: 

**==> picture [341 x 25] intentionally omitted <==**

**==> picture [377 x 14] intentionally omitted <==**

**==> picture [298 x 13] intentionally omitted <==**

_The Crossover Point._ The localized test dominates when E[ln _p_ local, final] _<_ E[ln _p_ global]: 

**==> picture [284 x 54] intentionally omitted <==**

This inequality demonstrates that localized detection is optimal when the signal is sufficiently concentrated (low _ρ_ ) such that the variance reduction from excluding human tokens outweighs the logarithmic search tax. 

## **E Additional Experiments and Details** 

## **E.1 Benchmark Variance Analysis** 

Evaluating LLM performance on benchmarks with chain-of-thoughts involves variance due to the stochastic generation, and the final answer extraction heuristics (e.g., regex-based parsing for numbers or letters, or code execution for programming tasks). To quantify this variance and assess whether watermarking introduces systematic degradation or improvement, we re-ran a subset of benchmarks with multiple random seeds for non-watermarked generation and multiple secret keys for watermarked generation. 

_Experimental Setup._ We use Qwen 3.5-27B with reasoning enabled (reasoning temperature 0.6, top- _p_ = 0 _._ 95, max 3,000 reasoning tokens). We evaluate on five benchmarks: AIME (math), GSM8K (math), HumanEval (code), MBPP (code), and MMLU (multiple choice). For non-watermarked generation, we use 5 different random seeds. For watermarked generation, we use the same values as secret keys, with both _n_ -gram deduplication enabled and disabled (see Remark 1): when enabled, watermark contexts that have already appeared in the generation fall back to vanilla sampling instead of watermarked sampling. 

_Results._ The standard deviation across seeds/keys ranges from 0.3% to 2.4%, depending on the benchmark and condition. Code benchmarks exhibit variance due to the binary nature of test execution and sensitivity to minor formatting differences. Crucially, the differences between watermarked and non-watermarked conditions fall within approximately one standard deviation, indicating no systematic performance degradation from watermarking. 

_Effect of n-Gram Deduplication._ Enabling _n_ -gram deduplication (falling back to vanilla sampling for repeated context windows) tends to produce lower variance, particularly visible on MMLU (0.6 vs 1.6 std) and HumanEval (0.7 vs 2.4 std). This is consistent with the observation that repeated _n_ -gram contexts in reasoning chains can lead to more deterministic (and potentially repetitive) generation patterns when not deduplicated. 

35 

**Table6** Benchmark accuracy (%) across 5 random seeds (non-watermarked) or 5 secret keys (watermarked). We report Mean _±_ Std to quantify generation variance. “Dedup” refers to _n_ -gram deduplication at generation-time (see Remark 1). Differences between conditions fall within one standard deviation, indicating no systematic degradation from watermarking. 

|Benchmark|No Watermark|WM (no dedup)|WM (dedup)|
|---|---|---|---|
|AIME|41_._0_±_1_._2|40_._6_±_1_._2|40_._8_±_0_._9|
|GSM8K|95_._9_±_0_._3|95_._6_±_0_._3|95_._9_±_0_._3|
|HumanEval|97_._1_±_0_._9|97_._6_±_2_._4|97_._8_±_0_._7|
|MBPP|50_._2_±_0_._6|49_._8_±_0_._4|49_._7_±_0_._6|
|MMLU|87_._6_±_0_._5|87_._1_±_1_._6|87_._8_±_0_._6|



**Table 7** Watermark detection performance on multilingual QA. 

|Metric|English|Arabic|Chinese|Hindi|Japanese|Overall|
|---|---|---|---|---|---|---|
|TPR@0.1%|53.6%|83.3%|79.5%|59.0%|51.0%|63.3%|
|Median log10_p_|_−_3_._15|_−_5_._23|_−_4_._62|_−_3_._51|_−_3_._04|_−_3_._72|



## **E.2 Multilingual QA** 

This section provides the full experimental setup for the multilingual question-answering evaluation. The same dataset and generation pipeline are used for both the watermark detection analysis below and the human preference evaluation in subsection E.3. 

_Experimental Configuration._ We use GPT-OSS-20B with reasoning enabled (max 2,000 reasoning tokens) and watermarking applied to the reasoning trace. Generation uses temperature 0 _._ 7, top- _p_ = 0 _._ 95, and a maximum length of 4,096 tokens. The watermark employs Gumbel-Max with 3-gram context, dual-key early fusion ( _α_ = 0 _._ 1), and a fixed secret key. 

_Datasets._ We evaluate on 6,000 question-answer pairs across five languages: English (2,000 samples from ELI5), and Arabic, Chinese, Hindi, and Japanese (1,000 samples each from CaLMQA (Arora et al., 2025)). 

_System Prompt._ The following system prompt was used for all languages: 

_“You are answering questions. Give a clear, concise explanation in plain language. Answer in the same language as the question. Keep your answer to 50–150 words. No bullet points, headers, or markdown formatting—just natural prose.”_ 

_Watermark Detection Results._ Arabic and Chinese show strongest detection, which is likely due to higher per-token entropy. Japanese shows lowest detection (51%) due to the more constrained vocabulary and lower entropy in CJK scripts. 

_Statistical Tests for Differences._ We apply McNemar’s test (McNemar, 1947) to assess whether watermarking systematically affects script consistency or refusal rates. For script consistency, we observe 52 discordant pairs where WM was wrong but Non-WM was correct, versus 39 where Non-WM was wrong but WM was correct; with continuity correction, this yields _χ_[2] = 1 _._ 58 and _p_ = 0 _._ 21. For refusal rates, we find 21 pairs where WM refused but Non-WM answered, versus 15 where Non-WM refused but WM answered, giving _χ_[2] = 0 _._ 69 and _p_ = 0 _._ 41. Both _p_ -values are well above the significance threshold ( _α_ = 0 _._ 05), indicating that watermarking does not seem to systematically increase script errors or refusals. 

## **E.3 Human Evaluation Details** 

This section provides methodology and detailed results for the human evaluation study summarized in subsection 4.4. The experimental setup (model, datasets, generation parameters) is shared with the 

36 

**Table 8 Full four-class preference breakdown** (majority vote, 3 annotators per sample). Split: items where no majority exists (three-way tie), counted as Tie in the final analysis. 

|**Language**|**N**|**Prefer WM**|**Prefer Base**|**Both Good**|**Both Bad**|**Split**|
|---|---|---|---|---|---|---|
|English|2,000|124|146|1,482|92|156|
|Arabic|1,000|201|181|168|287|163|
|Chinese|1,000|84|74|514|272|56|
|Hindi|1,000|91|89|435|278|107|
|Japanese|1,000|143|130|275|228|224|
|**Overall**|6,000|643|620|2,874|1,157|706|



multilingual QA experiment described in subsection E.2. 

_Preference Distribution._ Table 8 shows the complete four-class preference breakdown before merging tie categories. Annotators chose among: _A is preferred_ , _B is preferred_ , _Both equally good_ , and _Both equally bad_ . We aggregate via majority vote (at least 2/3 annotators agree); samples with a three-way split (one vote per distinct category) are assigned to “Tie.” For the final analysis, “Both Good,” “Both Bad,” and splits are merged into a single Tie category. 

_Net Win Rate._ We define the _net win rate_ as 

**==> picture [292 x 19] intentionally omitted <==**

where _n_ WM and _n_ Base are the number of samples where the watermarked or baseline response was preferred (by majority vote), and _N_ is the total number of samples including ties. The overall net win rate is +0 _._ 38%, indicating a negligible difference. 

_Binomial Test._ Among decisive (non-tie) samples, we test the null hypothesis _H_ 0 : _P_ (WM preferred) = 0 _._ 5 using a two-sided exact binomial test. No individual language reaches significance at _α_ = 0 _._ 05 (English: _p_ = 0 _._ 20; Arabic: _p_ = 0 _._ 33; Chinese: _p_ = 0 _._ 47; Hindi: _p_ = 0 _._ 94; Japanese: _p_ = 0 _._ 47). The overall pooled test yields _p_ = 0 _._ 54, also non-significant. The overall preference is nearly evenly split (50.9% WM among decisive samples), indicating no quality degradation. 

_Equivalence Testing (TOST with Ties)._ To establish imperceptibility, rather than failing to detect a difference, we apply the Two One-Sided Tests (TOST) procedure (Schuirmann, 1987). We test: 

**==> picture [449 x 23] intentionally omitted <==**

where proportions are computed over _all N_ samples (including ties in the denominator). This formulation is more powerful than restricting to decisive samples, because ties represent direct evidence of imperceptibility (the annotator could not distinguish between outputs) and contribute to the sample size. 

ˆ ˆ ˆ Let _d_[ˆ] = _p_ WM _− p_ Base with standard error SE = ~~�~~ (ˆ _p_ WM + _p_ Base _− d_[ˆ][2] ) _/N_ . The TOST procedure computes two one-sided _z_ -tests: _z_ 1 = ( _d_[ˆ] _−_ ∆) _/_ SE and _z_ 2 = ( _d_[ˆ] + ∆) _/_ SE, and rejects _H_ 0 when max(Φ( _z_ 1) _,_ 1 _−_ Φ( _z_ 2)) _< α_ . 

Table 9 reports results for ∆= 5%. Equivalence is established for four of five languages and overall ( _p <_ 0 _._ 05); Arabic marginally fails ( _p_ = 0 _._ 062) as its upper confidence bound (5 _._ 2%) slightly exceeds the _±_ 5 percentage-point margin. 

_On Trinomial Tests._ An alternative approach is the trinomial test for paired data with ties (Bian et al., 2011; Dathathri et al., 2024), which models the three-category distribution (WM, Base, Tie) directly. We experimented with this approach but found that the chi-square statistic converges rapidly with the number of ties: once more than a handful of ties are present, the _p_ -value stabilizes to the second decimal place and equals the standard binomial test on decisive samples. Since 79% of our 

37 

**Table 9 TOST equivalence test results** (∆= 5%, _α_ = 0 _._ 05). Proportions computed over all _N_ samples. 90% CI: Wald interval for the difference _P_ (WM) _− P_ (Base). 

|**Language**|_N_|ˆ_d_|90%|CI|_p_TOST|Result|
|---|---|---|---|---|---|---|
|English|2,000|_−_1_._10%|[_−_2_._5%_,_|+0_._3%]|_<_0_._001|Equivalent|
|Arabic|1,000|+2_._00%|[_−_1_._2%_,_|+5_._2%]|0_._062|Marginal|
|Chinese|1,000|+1_._00%|[_−_1_._1%_,_|+3_._1%]|_<_0_._001|Equivalent|
|Hindi|1,000|+0_._20%|[_−_2_._0%_,_|+2_._4%]|_<_0_._001|Equivalent|
|Japanese|1,000|+1_._30%|[_−_1_._4%_,_|+4_._0%]|0_._013|Equivalent|
|**Overall**|6,000|+0_._38%|[_−_0_._6%_,_|+1_._4%]|_<_0_._001|Equivalent|



**Table 10 Inter-annotator agreement statistics by language** (four-class scale). 

|**Language**|**Unanimous Rate**|**Majority (**_≥_**2/3)**|**Pairwise Agreement**|
|---|---|---|---|
|English|54.0%|92.2%|0.667|
|Arabic|23.9%|83.7%|0.438|
|Chinese|48.5%|94.4%|0.638|
|Hindi|37.0%|89.3%|0.544|
|Japanese|17.0%|77.6%|0.372|
|**Overall**|39.1%|88.2%|0.555|



samples are ties, the trinomial test provides no additional discriminative power, which motivates our use of the TOST procedure that explicitly leverages ties as evidence of imperceptibility. 

_Inter-Annotator Agreement._ We measure agreement using two metrics: (i) unanimous agreement rate (fraction of samples where all 3 annotators selected the same four-class option), and (ii) mean pairwise agreement (average fraction of annotator pairs that agree on the four-class label). Table 10 shows that agreement varies by language, with English and Chinese exhibiting the highest consistency. The lower agreement rates for Arabic and Japanese may reflect the inherent subjectivity of quality judgments and cultural differences in evaluation norms. 

## **E.4 Learnability Experimental Details** 

_Models & Dataset._ The teacher is DeepSeek-R1-Distill-Qwen-14B (Guo et al., 2025), an R1-style reasoning model that produces long chain-of-thought traces enclosed in <think> tags, and the student is Qwen2.5-3B (Team, 2024). We train on a subset of 5,000 problems drawn from OpenR1-Math220k (Hugging Face, 2025), curated via a three-stage pipeline: (i) malformed or incomplete problems are removed; (ii) only problems that the student model fails to solve are retained, ensuring the training data teaches new capabilities; (iii) diversity sampling across 14 math categories with a 15% cap per category prevents overrepresentation of any single topic. 

_Watermarked Trace Generation._ We compare the three sampling-based methods introduced in subsection 4.1: _Gumbel-Max_ (Aaronson and Kirchner, 2023), _TextSeal_ (dual-key routing probability _α_ = 0 _._ 1), and _SynthID_ (Dathathri et al., 2024) (plus an unwatermarked control), all with watermark context window _k_ = 3. Secret keys are calibrated per method via a Kolmogorov–Smirnov test to ensure uniform PRF hashes on unwatermarked text as done in Fernandez et al. (2025). The teacher generates 5,000 solutions using vLLM (Kwon et al., 2023) with flash-attention-2 on 4 _×_ H200 GPUs (tensor parallel), with _T_ = 1 _._ 0, top- _p_ = 0 _._ 95, and max 8,192 generated tokens. 

_Quality Filtering._ Each teacher trace passes through four sequential filters (the first failure rejects the trace): (i) _think closure_ —the trace must contain a closing </think> tag; (ii) _boxed presence_ —the trace must include a \boxed{...} final-answer pattern (skipped for multiple-choice datasets); (iii) _repetition detection_ —a sampled sliding-window check (window size 100 characters, _∼_ 200 evenly spaced samples) rejects any trace in which a substring occurs _≥_ 3 times (responses _≤_ 200 characters auto-pass); 

38 

**Table 11 Full learnability statistics (OpenR1,** _N_ =5 _,_ 000 **prompts).** Teacher _−_ log10( _p_ ) is the detection power of the watermark in the teacher’s own traces (mean and median across individual traces). Student _−_ log10( _p_ ) is the pooled detection power after distillation (original setting, all retained traces).[†] TextSeal uses entropy-weighted scoring. 

|**Method**|**Retained**|**Pass**|**Teacher**|_−_log10(_p_)|**Student**|
|---|---|---|---|---|---|
||**Traces**|**Rate**|Mean|Median|_−_log10(_p_)|
|Gumbel-Max|1,991|39.8%|14.89|9.09|24.80|
|TextSeal|2,352|47.0%|33.15†|27.50†|35.82†|
|SynthID|2,408|48.2%|14.39|12.12|13.54|
|Control|2,400|48.0%|0.39|0.25|—|



(iv) _answer verification_ —the extracted answer is compared to the gold answer using the math_verify library in a fail-open mode: if either side fails to parse, the trace is kept rather than rejected. 

_Student Fine-Tuning._ The student is fine-tuned on the filtered traces using LoRA (Hu et al., 2022) (rank 128, scaling factor 128, dropout 0.05) with learning rate 2 _×_ 10 _[−]_[5] and 3 epochs. The loss is computed over the full teacher response (both the reasoning trace and the final answer) while the prompt tokens are masked out. 

_Watermark Detection._ We evaluate watermark transfer using the _open-model_ radioactivity test of Sander et al. (2024, 2025). The test operates in a _teacher-forcing_ setup: each training trace is fed into the student model, and the student’s top-1 prediction ˆ _x_[(] _[t]_[)] = arg max _v∈V Pθ_ ( _v | x<t_ ) is recorded at every response position _t_ . Crucially, we score the student’s _predictions_ rather than newly generated text: this isolates the watermark signal from confounding factors such as sampling noise and generation quality, while requiring only a single forward pass over the existing traces rather than expensive autoregressive generation. If the student has internalized the watermark’s token preferences during fine-tuning, its top-1 predictions will be systematically biased toward high-PRF tokens—even without access to the secret key. 

We score each prediction using the watermark’s PRF: _Rt_ = PRF(ˆ _x_[(] _[t]_[)] _,_ **w** _t, K_ ), where **w** _t_ = ( _x_[(] _[t][−][k]_[)] _, . . . , x_[(] _[t][−]_[1)] ) is the trigram context window of _teacher_ tokens preceding position _t_ (as defined in section 2). Within each trace, each context window is scored only once; across traces, all (context, predicted token) pairs are pooled and deduplicated so that repeated tuples are counted only once, satisfying the independence assumption required by the statistical test (Fernandez et al., 2023). This yields _∼_ 1 _._ 4–2 _._ 2M unique scored tokens per method. 

For Gumbel-Max, a single pooled Gamma test produces the _p_ -value: we compute _st_ = _−_ ln(1 _− Rt_ ) for each unique pair and sum over all _n_ unique scored tokens to obtain _Sn_ =[�] _[n] t_ =1 _[s][t]_[,][which][under] _H_ 0 follows Γ( _n,_ 1) (Proposition 2). For TextSeal, we use the entropy-weighted early-fusion score with ˆ _wi_[ent] = � _Hi_ weighting (subsection 3.2), where entropy is estimated from the student model’s forward pass, and compute the _p_ -value via the moment-matched Gamma approximation of Equation 6; the choice of weighting function is validated by the ablation in Figure 9. For SynthID, we apply the frequentist test described in subsection 4.1, computing a depth-weighted Z-score over the tournament layers. 

_Teacher Trace Quality._ Pass rates are 48% for the control, 48.2% for SynthID, 47% for TextSeal, and 39.8% for Gumbel-Max, yielding 2,400, 2,408, 2,352, and 1,991 well-formed traces respectively. Gumbel-Max traces are also notably shorter on average ( _∼_ 2 _,_ 400 response tokens vs. _∼_ 3 _,_ 300–3 _,_ 500 for other methods), because its deterministic argmax selection causes more repetition loops at _T_ =1 _._ 0; the filter removes these long repetitive traces, leaving only shorter clean ones. As a result, the student is fine-tuned on different amounts of data across configurations; we do not normalize for this, as the variation in sample count is modest ( _∼_ 20%). 

_Controlled Comparisons._ The results above are obtained with each method’s full set of well-formed traces, which differ in count (1 _,_ 991–2 _,_ 408) and average length (Gumbel-Max traces average _∼_ 2 _,_ 400 

39 

tokens vs. _∼_ 3 _,_ 300–3 _,_ 500 for other methods). To rule out training data volume as a confound, we repeat the experiment under two controlled conditions (Figure 8): (i) _equal traces_ , where each method uses exactly 1 _,_ 991 traces (the Gumbel-Max minimum, randomly subsampled for the other methods), and (ii) _equal tokens_ , where each method is allocated _∼_ 15 _._ 1M characters (subsampling traces for methods with more tokens, using all available traces for Gumbel-Max). Under equal traces, TextSeal achieves the highest student accuracy (81 _._ 0%), followed by SynthID and Control (78 _._ 8% each) and Gumbel-Max (77 _._ 7%). Under equal tokens, the spread narrows (79 _._ 7%/78 _._ 6%/79 _._ 6%/77 _._ 6% for TextSeal/GumbelMax/SynthID/Control). In both settings all watermarked students substantially improve over the pre-training baseline (64 _._ 5%). Detection results confirm that all three watermarks remain strongly detectable under both controls, validating that the learnability conclusions of Figure 8 are not artifacts of unequal training data volume. 

_Entropy Weighting Ablation._ Each weighting variant in Figure 9 computes the weighted statistic _S_ combined =[�] _[n] i_ =1 _[w] i_[ent] _· si_ , where _si_ is TextSeal’s early-fusion score (Equation 3) and _wi_[ent] = _f_ ( _Hi_ ) is a function of the local entropy _Hi_ at position _i_ , estimated via a single forward pass of the student model. The _p_ -value is computed via the moment-matched Gamma approximation of Equation 6, which accounts for the heterogeneous weights. Concave normalized-entropy transforms outperform linear/superlinear alternatives because they moderately upweight high-entropy positions—where the watermark has more room to influence token selection (Proposition 3)—without over-amplifying noisy extreme-entropy tokens. Unnormalized power functions ( _Hi_[1] _[.]_[0] , _Hi_[1] _[.]_[5] ) are sensitive to the absolute entropy scale and perform no better than the unweighted baseline. 

## **F Extended Related Work** 

## **F.1 Post-Hoc Text Watermarking** 

Early text watermarking altered surface-level text characteristics such as characters or spacing (Brassil et al., 1995). Later methods modify grammatical or syntactical structures via pre-established rules (Topkara et al., 2005), including synonym substitution (Topkara et al., 2006c) and word reordering through passivization or topicalization (Topkara et al., 2006b,a; Meral et al., 2009). Text steganography follows similar principles (Winstein, 1998; Chapman et al., 2001; Bolshakov, 2004; Shirali-Shahreza and Shirali-Shahreza, 2008; Chang and Clark, 2014; Xiang et al., 2017). These edit-based systems exhibit low robustness and payload, e.g., 1–2 bits per sentence (Wilson and Ker, 2016). Deep learning methods have since been applied, including masked language models for steganography (Ueoka et al., 2021), infilling models (Yoo et al., 2023), neural lexical substitution (Qiang et al., 2023), and encoderdecoders (Abdelnabi and Fritz, 2021; Zhang et al., 2024; Xu et al., 2024). 

## **F.2 Generation-Time LLM Watermarking** 

The first watermarks for machine-generated text date back to a method presumably used in Google Translate to filter translations from future training data (Venugopal et al., 2011). For LLM-generated text, two concurrent approaches appeared shortly after the release of ChatGPT: Kirchenbauer et al. (2023a) bias a subset of the vocabulary (“green-red list”), while Aaronson and Kirchner (2023) alter the sampling via the Gumbel-max trick. Both use pseudorandom seeds generated from a secret key and preceding tokens, enabling lightweight statistical detection without access to the model. 

Subsequent work explores improved tests and multi-bit watermarking (Fernandez et al., 2023; Yoo et al., 2024; Qu et al., 2024), position-dependent seeds (Christ et al., 2023; Kuditipudi et al., 2023), low-entropy optimizations (Lee et al., 2023; Christ et al., 2023; Huang et al., 2023), and semantic watermarks for improved robustness (Liu et al., 2023; Liu and Bu, 2024; Fu et al., 2024; Hou et al., 2023, 2024). A key distinction is whether a method is _distortion-free_ : at each generation step, the nexttoken distribution is preserved, i.e., P(output _t_ = _v_ ) = _pv_[(] _[t]_[)] for all _v_ , where the probability is taken over the randomness of the watermark scheme (PRF seeds and, for dual-key methods, the key selection). Each individual token is drawn from the unmodified LLM distribution; only _diversity_ across repeated generations for the same prompt is reduced. See subsection A.3 for detailed scheme descriptions. Greenred list methods (Kirchenbauer et al., 2023a) and low-entropy filtering methods (e.g., SWEET (Lee 

40 

et al., 2023), which skips watermarking on low-entropy tokens) are _not_ distortion-free: they alter the output distribution, degrading every generation. MorphMark (Wang et al., 2025) adaptively scales the green-red bias based on the natural green-list probability mass, reducing distortion in low-entropy contexts, but remains non-distortion-free since it still applies a logit bias. Semantic watermarks (Liu et al., 2023; Liu and Bu, 2024; Hou et al., 2023) require auxiliary semantic encoders at generation time, making them harder to deploy. Gumbel-max (Aaronson and Kirchner, 2023), Permute-and-Flip (Zhao et al., 2024), DiPMark (Wu et al., 2023) (distortion-free green-red via pseudorandom permutations), SynthID-Text (Dathathri et al., 2024) (deployed in Google Gemini via tournament-based sampling), and WaterMax (Giboulot and Furon, 2024) (multiple generations per query, impractical for production) are distortion-free. Toolkits have also been introduced to benchmark these methods (Piet et al., 2023; Pan et al., 2024). Recent large-scale evaluations (Fernandez et al., 2025) show that Gumbel-max and SynthID achieve the best detectability-quality Pareto frontier among all methods, strictly dominating DiPMark, green-red variants, and semantic watermarks. 

TextSeal builds on the Gumbel-max framework but introduces dual-key generation for diversity, entropyweighted detection, and localized multi-region search—none of which are present in prior work. We therefore compare TextSeal against these two practical baselines. Because all three are distortion-free, the comparison is controlled: we fix the LLM, temperature, and top- _p_ , and only vary the watermarkspecific diversity parameter (key routing probability _α_ for TextSeal, tournament depth for SynthID), isolating the watermark’s effect from the decoding strategy. 

## **F.3 Post-Hoc LLM Watermarks for Data Protection** 

Recent works apply LLM watermarks to training or evaluation data via paraphrasing. Most exploit watermark radioactivity (Sander et al., 2024), i.e., the detectable traces left when watermarked text is used for training. Applications include detection of texts used in retrieval-augmented generation (Jovanović et al., 2025), benchmark contamination detection (Sander et al., 2025), and training data copyright (Zhang et al., 2025). Waterfall (Lau et al., 2024) evaluates post-hoc watermarking through LLM paraphrasing for provenance on code and natural text. In section 6, we demonstrate that TextSeal’s watermark transfers through distillation, extending this line of work to reasoning trace provenance. 

41 

