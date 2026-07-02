## **Task 1** 

# **Text Watermark Localization** 

## **Language Models: Autoregressive Predictions** 

This 

is 

**==> picture [88 x 30] intentionally omitted <==**

**----- Start of picture text -----**<br>
LLM<br>**----- End of picture text -----**<br>


2 CISPA Championship 

## **Language Models: Autoregressive Predictions** 

## This is 

**==> picture [503 x 174] intentionally omitted <==**

**----- Start of picture text -----**<br>
fine: 0.10<br>great: 0.04<br>nice: 0.06<br>fine.<br>LLM predicts the token with<br>LLM<br>the highest likelihood<br>**----- End of picture text -----**<br>


**Vocabulary** 

3 CISPA Championship 

## **Language Model Watermarking: Red-Green List** 

## This is 

**==> picture [250 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
fine: 0.10<br>great: 0.04<br>nice: 0.06<br>_—_—<br>LLM<br>**----- End of picture text -----**<br>


[Internetfor 99.999999999% fine. of the Synthetic 

**Intuition:** Divide the vocabulary in “red” and “green” tokens at random **Normal (non-watermarked) text:** Red and green tokens are equally likely **Watermarked text:** Green tokens are more likely 

[Kirchenbauer et al., 2023] 

4 CISPA Championship 

## **Language Model Watermarking: Red-Green List** 

fine: 0.10 great: 0.04 Get the probability vector nice: 0.06 

Use previous token to generate random number 

Use the seed to split vocab into red & green lists Sample tokens from green list 

Perform statistical testing: are green tokens in the text significantly more likely than random chance? 

yes Watermarked? no 

[Kirchenbauer et al., 2023] 

5 CISPA Championship 

## **Task 1: Text Watermark Localization** 

## **Identify Watermarked Text Regions:** 

Stopped 30 of 34 shots in the Blue Jackets’ 6-2 win over the Canadiens . on Monday The loss dropped him to 30-22-1 on the season with a 2.76 GAA and.906 The victory pushed him to 31-21-1 on the season with a 2.72 GAA and.908 save percentage.\nMar. 10 11:49 AM PT12:49 PM [...continues] 

– **non** -watermarked text loss dropped … The victory … – **watermarked** text 

6 CISPA Championship 

## **Task 1: Text Watermark Localization** 

## **Identify Watermarked Text Regions:** 

Stopped 30 of 34 shots in the Blue Jackets’ 6-2 win over the Canadiens . on Monday The loss dropped him to 30-22-1 on the season with a 2.76 GAA and.906 The victory pushed him to 31-21-1 on the season with a 2.72 GAA and.908 save percentage.\nMar. 10 11:49 AM PT12:49 PM [...continues] 

– **non** -watermarked text loss dropped … The victory … – **watermarked** text 

**Watermarking schemes fully provided:** 1. TextSeal 

2. Gumbel-Max 

3. Unigram 

4. KGW (red-green watermark) 

**Non-watermarked text:** 

1. Human-created 

2. Non-watermark generation 

3. Different watermarks than the above 

7 CISPA Championship 

## **Task 1: Score for the Text Watermark Localization** 

## **Identify Watermarked Text Regions:** 

Stopped 30 of 34 shots in the Blue Jackets’ 6-2 win over the Canadiens . on Monday The loss dropped him to 30-22-1 on the season with a 2.76 GAA and.906 The victory pushed him to 31-21-1 on the season with a 2.72 GAA and.908 save percentage.\nMar. 10 11:49 AM PT12:49 PM [...continues] 

**Assign a continuous Score in range [0,1] to each token:** 

_Stopped:_ 0.8321 _30:_ 0.722 

…. 

**Score** ≥ **0.5** : token is from a watermarked text region **Score** < **0.5** : token is clean 

8 CISPA Championship 

## **Task 1: Score for the Text Watermark Localization** 

## **Identify Watermarked Text Regions:** 

Assign a continuous Score in range [0,1] to each token: 

**Score** ≥ **0.5** : token is from a watermarked text region **Score** < **0.5** : token is clean 

# **The final score across all the tokens: TPR @ 0.1% FPR** 

9 CISPA Championship 

## **Task 1: Text Watermark Localization** 

## **Data:** 

## 1500 text documents: 

Training set: 90 documents 

Validation set: 90 documents Test set: 1320 documents 

10 CISPA Championship 

## **Task 1: Text Watermark Localization** 

Training/Validation record: 

## **Data:** 

## 1500 text documents: 

Training set: 90 documents Validation set: 90 documents Test set: 1320 documents 

{ "document_id": "train_1", "text": ”Paul’s statement", "token_ids": [25300, 594, 5114], "token_pieces": ["Paul", "'s", ”statement"], "labels": [0, 0, 1] } 

11 CISPA Championship 

## **Task 1: Text Watermark Localization** 

## **Data:** 

## 1500 text documents: 

Training set: 90 documents Validation set: 90 documents Test set: 1320 documents 

Test record: { "document_id": "train_1", "text": ”Paul’s statement", "token_ids": [25300, 594, 5114], "token_pieces": ["Paul", "'s", ”statement"] } 

12 CISPA Championship 

## **Task 1: Text Watermark Localization** 

## **Submission format (example):** 

{"document_id": "1", "scores": [0.02, 0.04, 0.91, 0.87, …]} {"document_id": "2", "scores": [0.12, 0.16, 0.21, …]} … 

13 CISPA Championship 

## **Task 2** 

# **Member vs Generated Inference (MGI)** 

14 CISPA Championship 

## **Our Large Data Leads to Impressive Capabilities** 

Generative Models: **GPT** … **LLaMA** 

**==> picture [484 x 23] intentionally omitted <==**

**----- Start of picture text -----**<br>
Train Generate<br>**----- End of picture text -----**<br>


**==> picture [102 x 37] intentionally omitted <==**

**----- Start of picture text -----**<br>
import sys # system<br>print(“Data Provenance<br>for Generative AI”)<br>**----- End of picture text -----**<br>


15 CISPA Championship 

2 

## **Generative Models Memorize Training Data** 

**==> picture [407 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
Train Generate<br>**----- End of picture text -----**<br>


## **Copyright & Privacy Violations** 

> 100 lawsuits, 100 bn € 

16 CISPA Championship 

3 

## **Generative Models Produce Misinformation** 

**==> picture [407 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
Train Generate<br>**----- End of picture text -----**<br>


**Copyright & Privacy Violations** 

**Fake Data** 

17 CISPA Championship 

3 

## **Generative Models Create Data Loops** 

**Copyright & Privacy Train Data Loops Generate Violations Quality Degradation** —> = i i, 

**Fake Data** 

CISPA Championship 

18 

3 

## **Task 2: Member vs Generated Inference (MGI)** 

**==> picture [543 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
, …<br>𝑀 𝐺 𝐺′𝑀<br>Enco DecoEnco Deco<br>de de<br>de de<br>𝐺 𝐺′<br>𝑁 𝑁<br>𝑁<br>**----- End of picture text -----**<br>


𝑀 – Member training samples for model 

𝑁 – Non-member data from the same distribution as data 𝑀 

𝐺 – Generated data from model 

CISPA Championship 

19 

3 

**Task 2: Member vs Generated Inference (MGI)** ll My l[ <=> | 𝑀 𝐺 **Data** = Enco S— & mee _.x. 𝑀, 𝑁, or 𝐺 **Detector** : Data de Classifier for ll 𝑀, 𝑁, or 𝐺 <P ll 𝑁 wee —> l **Fool the Detector to misclassify:** 

𝑀→𝑁 𝑀→𝐺 𝑁→𝑀 𝑁→𝐺 𝐺→𝑀 𝐺→𝑁 

20 CISPA Championship 

3 

**Task 2: Member vs Generated Inference (MGI)** ll My l[ <=> | 𝑀 𝐺 **Data** = Enco S— & mee _.x. 𝑀, 𝑁, or 𝐺 **Detector** : Data de Classifier for ll 𝑀, 𝑁, or 𝐺 <P ll 𝑁 wee —> l **Fool the Detector to misclassify:** 

**You are given:** 𝑀 – 300 samples 𝑵 – 300 samples 𝑮 – 300 samples 

𝑀→𝑁 𝑀→𝐺 𝑁→𝑀 𝑁→𝐺 𝐺→𝑀 𝐺→𝑁 

21 CISPA Championship 

3 

## **Task 2: Member vs Generated Inference (MGI)** 

**The submission format:** 𝑀→𝑁 (300 examples starting from the 𝑀 set) 𝑀→𝐺 (300 examples starting from the 𝑀 set) 𝑁→𝑀 (300 examples starting from the 𝑁 set) 𝑁→𝐺 (300 examples starting from the 𝑁 set) 𝐺→𝑀 (300 examples starting from the 𝐺 set) 𝐺→𝑁 (300 examples starting from the 𝐺 set) 

22 CISPA Championship 

## **Task 2: Member vs Generated Inference (MGI)** 

## **Assessment metrics:** 

1. 𝑫𝒆𝒕𝒆𝒄𝒕𝑺𝒄𝒐𝒓𝒆= 𝟏 if sample misclassified and 𝟎 otherwise. 

**Data** 

**Detector** : Data Classifier for 𝑀, 𝑁, or 𝐺 

𝑀, 𝑁, or 𝐺 

## **Fool the Detector to misclassify:** 

𝟏[𝟐] = 2. **i** 𝑴𝑺𝑬 𝐗−𝑿[′] Image -th Quality: 𝒊 , 𝒎 where **m** is number of pixels, 𝑿- the initial image, 𝑿[′] - your modified image. The lower 𝑴𝑺𝑬, the better. 

𝑀→𝑁 

𝑀→𝐺 

𝑁→𝑀 𝑁→𝐺 𝐺→𝑀 

𝐺→𝑁 

3. You have to return **n=1800** images, for which the final 𝒔𝒄𝒐𝒓𝒆∈[𝟎, 𝟏] is: 𝒏 𝟏 𝒔𝒄𝒐𝒓𝒆= 𝑫𝒆𝒕𝒆𝒄𝒕𝑺𝒄𝒐𝒓𝒆× (𝟏−𝑴𝑺𝑬𝒊) 𝒏[෍] 𝒊=𝟏 

23 CISPA Championship 

## **Task 3** 

# **Data Reconstruction from Gradients** 

24 CISPA Championship 

## **Individuals Generate Sensitive Data** 

**==> picture [75 x 24] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alice<br>**----- End of picture text -----**<br>


25 

## **Companies Train Machine Learning Models** 

**==> picture [75 x 24] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alice<br>**----- End of picture text -----**<br>


26 

**Centralized vs Federated Learning** Central Server NO) Go **Gradients** — SB a6e a ate Central Server Server has Alice’s data 

… 

Alice 

**Centralized Learning** 

**Federated Learning** 

Alice 

27 

27 

• 

## **Alice’s Privacy Relies Purely on the Gradients** 

**==> picture [872 x 339] intentionally omitted <==**

**----- Start of picture text -----**<br>
Central Server<br>Should hide<br>-_<br>Alice’s data<br>Shared Model ae<br>Gradients Gradients<br>Gradients Gradients<br>SQ<br>Gradients Gradients<br>Alice<br>M Users<br>**----- End of picture text -----**<br>


28 CISPA Championship 

**Task 3: Reconstruct Data from Gradients** Honest-but-Curious Server (oie) ™“ Alice You are a passive, honest-but-curious server who wants to extract a significant number of sensitive user photos from their returned gradients.[“e*] 29 CISPA Championship 

## **Task 3: Setup with Shared Models and Gradients** 

Model 1 

… Model 2 

Model 12 

… **Gradients Gradients** 

**Gradients** 

128 photos 

128 photos 

128 photos 

## **Task 3: Submission Format and Score** 

. Submit a .pt file with **n=1536 reconstructed images** Provide 128 images each with dimension (3, 64, 64) per each of 12 models. Each pixel is float32 in [0,1]. Format of the submission: 

{ “model1”: Tensor(128, 3, 64, 64) “model2”: Tensor(128, 3, 64, 64) 

… 

} 

𝟏 **Score:** 𝒏 𝑺𝑺𝑰𝑴 𝒊 𝒏[σ][𝒊=𝟏] 

31 CISPA Championship 

