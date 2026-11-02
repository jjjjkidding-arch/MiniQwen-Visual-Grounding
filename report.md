# MiniQwen: Investigating Visual Grounding
in a Small Vision-Language Model

## Abstract

## 1. Introduction
Vision-language models (VLMs) combine visual and textual information to perform tasks such as image captioning and visual question answering. However, generating plausible descriptions does not necessarily imply that a model accurately understands or relies on the visual information provided.

This investigation examines whether a relatively small local vision-language model can learn meaningful visual grounding through a custom visual encoder and a pretrained language model. In particular, it investigates whether the model’s caption predictions depend on the input image, rather than primarily on linguistic patterns learned during pretraining.

To investigate this question, a custom CNN visual encoder and an MLP projector are connected to Qwen2.5-1.5B-Instruct, with LoRA used for parameter-efficient adaptation. The model is trained on the Flickr8k image-caption dataset using teacher-forced caption prediction.

Several controlled experiments are conducted, including small-dataset overfitting, correct-versus-mismatched image comparisons, zero-image ablation, and pairwise ranking-loss training. These experiments aim to distinguish the ability to predict captions from the ability to condition those predictions on the corresponding visual inputs.

The main objective is to identify where visual grounding succeeds or fails in this small-scale architecture and to examine whether additional training objectives improve the model’s sensitivity to visual information.

In this case, this vision-language models combine visual and textual representations to perform multimodal tasks. Contrastive approaches such as CLIP [1] have demonstrated the effectiveness of learning transferable visual representations from image-text pairs.

## 2. Model Architecture
This investigation uses the pretrained Qwen2.5-1.5B-Instruct language model [4], adapted through Low-Rank Adaptation (LoRA) [3] applied to the query and value projection matrices in its attention layers.
A custom convolutional neural network (CNN) serves as the visual encoder, extracting a 256-dimensional feature vector from each input image. A two-layer multilayer perceptron (MLP) projector then maps the visual features into the 1536-dimensional embedding space of Qwen.
The resulting visual embeddings are concatenated with the text prompt embeddings and passed into the language model for caption generation.

## 3. Experimental Setup
3.1 Training Objective

The model is trained using teacher-forced caption prediction with cross-entropy loss. During training, the ground-truth caption tokens are provided as input, and the model learns to predict each subsequent token conditioned on the visual embeddings and preceding caption tokens.

The pretrained Qwen parameters remain frozen, while the visual encoder, projector, and LoRA adapters are optimized.

## 4. Evaluation Methodology
4.1 Visual Input Ablation

To evaluate whether the model’s caption predictions depend on visual information, three input conditions are compared:

1. Correct image: The image corresponding to the ground-truth caption is provided.
2. Wrong image: The corresponding image is replaced with a mismatched image from another sample.
3. Zero image: The visual embeddings are replaced with zeros, removing the image-specific visual information.

For each condition, teacher-forced cross-entropy loss is calculated against the same ground-truth caption.

A lower loss under the correct-image condition suggests that the model assigns higher probability to the reference caption when provided with the corresponding visual input.

4.2 Loss Gap and Win Rate


Two loss differences are calculated to quantify sensitivity
to visual information.

**Wrong–Correct Gap:**

$$
\Delta_{\text{wrong}} =
L_{\text{wrong}} - L_{\text{correct}}
$$

**Zero–Correct Gap:**

$$
\Delta_{\text{zero}} =
L_{\text{zero}} - L_{\text{correct}}
$$

Positive gaps indicate that the correct image produces a
lower caption prediction loss than the mismatched or
zero-image conditions.

Additionally, the win rate measures the proportion of
evaluated samples for which the correct image produces
a lower loss than the mismatched image:

$$
\text{Win Rate} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{1}
\left(
L_{\text{correct},i} < L_{\text{wrong},i}
\right)
$$

A higher win rate indicates more consistent discrimination between corresponding and mismatched images.

4.3 Qualitative Caption Generation

In addition to teacher-forced evaluation, captions are generated autoregressively under different visual input conditions.

The generated captions are compared to examine whether changing or removing the image produces meaningful changes in the model’s descriptions.

This evaluation complements the loss-based metrics, since lower teacher-forced loss does not necessarily guarantee accurate or visually grounded free-form generation.

4.4 Evaluation Protocol

Evaluation is conducted on image-caption pairs using the same model checkpoint across the three visual input conditions.

For each sample, the ground-truth caption and text prompt remain unchanged, while the visual input is modified according to the correct-image, mismatched-image, or zero-image condition.

Teacher-forced cross-entropy loss is computed for each condition, and the resulting losses are aggregated to calculate mean loss, loss gaps, and win rate.

The evaluation is performed on both small-scale overfitting experiments and larger-scale training checkpoints to investigate whether visual discrimination generalizes beyond the training examples.

For reproducibility, the evaluation sample size, image-mismatch selection procedure, random seed, and checkpoint identity are recorded for each experiment.
## 5. Results

### 5.1 Small-Set Overfitting Experiment
The model was initially evaluated on a small dataset of 30 images to determine whether the architecture could learn associations between visual inputs and their corresponding captions.

The experiment demonstrated strong discrimination between correct and mismatched visual inputs under teacher-forced evaluation.

| Metric | Value |
|---|---:|
| Evaluation samples | 30 |
| Correct-image loss | 0.0827 |
| Wrong-image loss | 1.4213 |
| Zero-image loss | 2.3218 |
| Wrong–Correct gap | 1.3386 |
| Zero–Correct gap | 2.2391 |
| Win rate | 100% |

The correct-image condition achieved substantially lower loss than both the mismatched-image and zero-image conditions. This indicates that the architecture can learn to use image-specific information in the small-scale overfitting setting.

However, these results alone do not establish that the model generalizes to unseen images due to small size.

### 5.2 Full-Dataset Training
5.2 Larger-Scale Training

Following the small-set overfitting experiment, the model was trained on a larger subset of Flickr8k containing 1,000 images over five epochs.

The training loss decreased from approximately 3.56 to 2.50 as shown, while the validation loss reached approximately 2.47, indicating improved caption prediction performance.
 
Metric | Value |
|---|---:|
| Dataset | Flickr8k |
| Training images | 1,000 |
| Training epochs | 5 |
| Initial training loss | ~3.56 |
| Final training loss | ~2.50 |
| Validation loss | ~2.47 |
| Caption generation | Predominantly generic |
| Image-swap generation | Similar captions across images |

However, qualitative generation tests revealed that the model frequently produced generic descriptions. Furthermore, replacing the correct input image with a mismatched image often resulted in similar generated captions.

These observations suggest that improved teacher-forced caption prediction did not necessarily correspond to reliable image-conditioned caption generation.


### 5.3 Controlled Image-Swap Evaluation

All three checkpoints were evaluated on the same 300 held-out Flickr8k image-caption pairs, using random seed 42 and identical mismatched-image assignments. Teacher-forced caption losses were measured under correct-image, wrong-image, and zero-image conditions.

| Metric | Original | VL-best | Ranking-best |
|---|---:|---:|---:|
| Evaluation samples | 300 | 300 | 300 |
| Correct-image loss | 2.43831 | 2.40460 | 2.41286 |
| Wrong-image loss | 2.47180 | 2.47625 | 2.50662 |
| Zero-image loss | 2.91019 | 2.87238 | 2.87874 |
| Wrong–Correct gap | 0.03349 | 0.07164 | 0.09375 |
| Zero–Correct gap | 0.47188 | 0.46777 | 0.46588 |
| Win rate | 66.3% | 68.7% | 67.7% |

The wrong–correct loss gap increased from 0.03349 in the original checkpoint to 0.07164 in VL-best and 0.09375 in ranking-best. However, the win rate did not increase consistently, reaching its highest value of 68.7% in VL-best. This indicates stronger average discrimination between correct and mismatched visual inputs, without a corresponding consistent improvement across individual samples.

### 5.4 Paired Bootstrap Analysis

A paired bootstrap analysis was conducted to assess differences in the mean wrong–correct loss gap across checkpoints, using the same evaluation samples.

| Checkpoint comparison | Gap improvement | 95% confidence interval |
|---|---:|---|
| VL-best − Original | +0.03815 | [0.02716, 0.04949] |
| Ranking-best − VL-best | +0.02211 | [0.00837, 0.03619] |
| Ranking-best − Original | +0.06027 | [0.04339, 0.07759] |

All three confidence intervals exclude zero, supporting an increase in the mean wrong–correct loss gap between successive checkpoints. However, these improvements in teacher-forced visual discrimination do not independently establish improvements in free-form caption accuracy.

### 5.5 Qualitative Caption Generation

### 5.5 Qualitative Caption Generation

To investigate whether improvements in teacher-forced visual discrimination translate into meaningful caption generation, autoregressive outputs from the three checkpoints were compared under correct-image, mismatched-image, and zero-image conditions.

The following examples compare correct-image and mismatched-image generations against the reference captions.

#### Example 1: White Dog Resting

**Reference caption:** "A closeup of a white dog that is laying its head on its paws."

| Checkpoint | Correct-image generation | Wrong-image generation |
|---|---|---|
| Original | A man in a red shirt and a woman in a black shirt are standing in front of a building. | A man in a red shirt and a woman in a black shirt are standing in front of a building. |
| VL-best | A man in a red shirt is standing on a rock. | A man in a red shirt is standing on a rock. |
| Ranking-best | A man is standing on a rock in the middle of a field. | A man in a red shirt is standing on a rock. |

None of the three checkpoints accurately described the reference image. The ranking-best checkpoint produced different captions under correct and mismatched visual inputs, but neither description matched the reference.

#### Example 2: Boy at the Beach

**Reference caption:** "A boy in his blue swim shorts at the beach."

| Checkpoint | Correct-image generation | Wrong-image generation |
|---|---|---|
| Original | A man in a red shirt and a woman in a black shirt are standing in front of a building. | A man in a red shirt and a woman in a black shirt are standing in front of a building. |
| VL-best | A man in a red shirt is standing on a rock. | A man in a red shirt is standing on a rock. |
| Ranking-best | A man in a red shirt is standing on a rock. | A man is standing on a rock in the middle of a field. |

The original and VL-best checkpoints generated identical captions despite image replacement. The ranking-best checkpoint showed some variation, although neither description matched the reference.

#### Example 3: Woman Waiting for a Ride

**Reference caption:** "A blond woman in a blue shirt appears to wait for a ride."

| Checkpoint | Correct-image generation | Wrong-image generation |
|---|---|---|
| Original | A man in a red shirt and a woman in a black shirt are standing in front of a building. | A man in a red shirt and a woman in a black shirt are standing in front of a building. |
| VL-best | A man in a red shirt and a woman in a black shirt are holding hands. | A man in a red shirt and a woman in a white shirt are holding hands. |
| Ranking-best | A man in a red shirt is holding a baby in his arms. | A man in a red shirt is holding a baby in a blue shirt. |

Both VL-best and ranking-best generated different descriptions under correct and mismatched inputs. However, neither checkpoint produced a caption that accurately matched the reference description.

#### Summary of Qualitative Findings

Across these three examples, the original checkpoint repeatedly generated identical descriptions regardless of the input image. The VL-best and ranking-best checkpoints exhibited greater variation in some cases, but their outputs remained largely inconsistent with the reference captions.

This illustrates that increased sensitivity to visual inputs, as measured by teacher-forced loss differences, does not necessarily translate into accurate free-form image captioning.


## 6. Discussion

## 6. Discussion

### 6.1 Small-Set Learning and Generalization

The 30-image overfitting experiment demonstrated that the architecture could learn strong associations between visual inputs and their corresponding captions. The correct-image condition achieved substantially lower teacher-forced loss than the mismatched-image and zero-image conditions, with a 100% win rate.

However, this performance did not generalize to the larger held-out evaluation. In the controlled 300-image comparison, the original checkpoint achieved a wrong–correct loss gap of only 0.03349 and a win rate of 66.3%.

This contrast suggests that the architecture possesses sufficient capacity to learn image-caption associations on a small training set, but struggles to generalize these associations to unseen images.

One possible explanation is the limited representational capacity of the custom CNN visual encoder. The encoder consists of four convolutional blocks with progressively increasing channel dimensions (32, 64, 128, and 256), followed by max pooling and global average pooling to produce a single 256-dimensional feature vector. Although this architecture can extract visual features, its relatively simple structure and training from scratch on a limited image-caption dataset may restrict the learning of transferable visual representations. Furthermore, global average pooling compresses spatial information, potentially discarding relationships between objects that could be useful for visual grounding. However, the present experiments do not isolate the visual encoder as the primary cause of the observed generalization gap.

### 6.2 Effects of Training Objectives on Visual Discrimination

The controlled evaluation revealed progressively larger wrong–correct loss gaps across the original, VL-best, and ranking-best checkpoints, increasing from 0.03349 to 0.07164 and 0.09375, respectively.

The paired bootstrap analysis further supported these differences, with confidence intervals for the mean gap improvements excluding zero.

These findings indicate that the later training configurations increased the model's average sensitivity to whether the correct image was provided.

However, the win rate did not improve consistently. VL-best achieved the highest win rate of 68.7%, compared with 67.7% for ranking-best.

This suggests that an increase in the mean loss gap does not necessarily correspond to improved discrimination on a larger proportion of individual samples. A stronger average gap may arise from larger improvements on some samples without equivalent improvements across the entire evaluation set.

Furthermore, because the checkpoints differ in their training histories, the observed differences cannot be attributed exclusively to the ranking objective without additional controlled ablation experiments.

### 6.3 Teacher-Forced Discrimination Versus Free-Form Generation

Despite improvements in teacher-forced visual discrimination, qualitative generation remained weak across all three checkpoints.

The original model frequently produced identical captions for different images. The VL-best and ranking-best checkpoints showed greater variation in some examples, but many generated descriptions remained inconsistent with the reference captions.

This reveals an important distinction between two capabilities:

1. Assigning higher probability to a reference caption when the correct image is provided.
2. Independently generating an accurate caption from the visual input.

Teacher-forced evaluation supplies the ground-truth caption prefix during prediction. Consequently, the model can achieve lower loss without necessarily learning to generate accurate descriptions autoregressively.

During free-form generation, the model must rely on its own previously generated tokens. Generic or inaccurate early predictions may influence subsequent tokens, potentially producing repetitive descriptions.

The results therefore suggest that visual sensitivity measured through conditional likelihood is not sufficient evidence of successful visual grounding in generation.

### 6.4 Implications for Small Vision-Language Models

The experiments demonstrate that a relatively small vision-language architecture can learn image-dependent caption probabilities, particularly in a controlled overfitting setting.

However, the substantial difference between small-set discrimination and held-out caption generation highlights the difficulty of learning transferable visual grounding from limited image-caption data.

The ranking-based training objective increased the average correct-versus-mismatched image loss gap, but did not establish reliable free-form caption accuracy.

These findings suggest that future improvements should be evaluated along multiple dimensions, including visual discrimination, generalization to unseen images, and semantic correctness of generated descriptions.

In particular, stronger pretrained visual representations, more targeted grounding tasks, and controlled comparisons of training objectives may help identify which components limit performance.

## 7. Limitations and Future Work
The present investigation has several limitations.

First, the custom CNN visual encoder consists of four convolutional blocks trained from scratch. Its relatively simple architecture and global average pooling may limit the extraction of transferable semantic features and the preservation of spatial information.

Second, visual-language integration relies on a two-layer MLP projector that maps image features into the language model’s embedding space. This approach does not explicitly preserve object-level representations or spatial relationships, potentially restricting the information available for language generation.

Third, the training dataset and computational resources were limited. Although Flickr8k provides paired image-caption examples, its relatively small scale may constrain the model’s ability to generalize to diverse unseen scenes.

Finally, the evaluation primarily relies on teacher-forced caption losses, image-swap comparisons, and selected qualitative examples. These measurements provide evidence of image sensitivity but do not comprehensively establish semantic correctness or reliable visual grounding.

Future work could investigate pretrained visual encoders, spatially structured visual tokens, improved multimodal alignment mechanisms, and larger-scale training. Additional evaluations using semantic caption metrics and controlled visual grounding tasks could help determine which architectural components most strongly affect performance.


## 8. Conclusion

This investigation examined visual grounding in a small vision-language model combining a custom CNN encoder, an MLP projector, and Qwen2.5-1.5B-Instruct with LoRA adaptation.

These experiments demonstrated strong image-dependent caption prediction in a small-set overfitting setting, but substantially weaker generalization to unseen images. Subsequent training checkpoints achieved larger correct-versus-mismatched image loss gaps, with the ranking-best checkpoint reaching 0.09375 compared with 0.03349 in the original model.

However, improvements in teacher-forced visual discrimination did not consistently amount to accurate autoregressive caption generation.

These findings highlight the distinction between sensitivity to visual inputs and reliable semantic grounding, motivating further investigation into visual representations, multimodal alignment, and evaluation methodology.




## References

[1] Radford et al. (2021).
*Learning Transferable Visual Models From Natural Language Supervision.*
[Paper](https://arxiv.org/abs/2103.00020)

[2] Liu et al. (2023).
*Visual Instruction Tuning.*
[Paper](https://arxiv.org/abs/2304.08485)

[3] Hu et al. (2022).
*LoRA: Low-Rank Adaptation of Large Language Models.*
[Paper](https://arxiv.org/abs/2106.09685)

[4] Yang et al. (2024).
*Qwen2.5 Technical Report.*
[Paper](https://arxiv.org/abs/2412.15115)

[5] Hodosh, Young, and Hockenmaier (2013).
*Framing Image Description as a Ranking Task:
Data, Models and Evaluation Metrics.*
[Paper](https://doi.org/10.1613/jair.3994)
