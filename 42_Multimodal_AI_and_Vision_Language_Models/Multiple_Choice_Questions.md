# Multiple Choice Questions: Multimodal AI and Vision-Language Models

📺 **Video Lecture:** https://youtu.be/GEmTIyLfVLU


## Question 1
Which fusion strategy in multimodal learning concatenates raw features from different modalities as input to a single model, enabling deep interactions but requiring aligned, synchronized data?

A) Late Fusion  
B) Early Fusion  
C) Hybrid Fusion  
D) Cross-Modal Fusion

---

## Question 2
In CLIP's contrastive learning approach, which of the following is treated as a negative pair during training?

A) An image and its matching caption  
B) An image and a caption from a different image in the batch  
C) Two images that describe the same object  
D) Text embeddings that are semantically similar

---

## Question 3
What is the primary advantage of CLIP's zero-shot classification capability?

A) It requires less training data than supervised models  
B) It enables classification on new categories without fine-tuning by leveraging learned image-text alignment  
C) It has built-in transfer learning from computer vision tasks  
D) It always outperforms supervised models on standard benchmarks

---

## Question 4
Which component in BLIP-2 acts as a lightweight adapter bridging frozen vision and language encoders?

A) Cross-Attention Module  
B) Vision Transformer (ViT)  
C) Q-Former  
D) Attention Mechanism

---

## Question 5
GPT-4V enables multimodal understanding by accepting images as tokens in the input sequence. Which of the following is NOT listed as a capability of GPT-4V?

A) Visual Question Answering  
B) Image Generation and Modification  
C) Document Understanding and Table Extraction  
D) Scene Understanding and Spatial Reasoning

---

## Question 6
In Visual Question Answering, what is a significant challenge when models are trained on datasets like VQA v2?

A) The lack of sufficient training data  
B) Models exploiting dataset biases and statistical shortcuts rather than achieving true understanding  
C) Insufficient computational resources for training  
D) Poor performance on unbalanced datasets

---

## Question 7
Which training technique in image captioning optimizes for caption-level metrics like BLEU and METEOR instead of token-level cross-entropy loss?

A) Beam Search  
B) Teacher Forcing  
C) Self-Critical Training  
D) Attention Mechanism Training

---

## Question 8
Stable Diffusion differs from DALL-E 2 primarily by:

A) Using pixel space instead of latent space for diffusion  
B) Diffusing in a learned latent space, enabling faster inference than DALL-E 2  
C) Generating only black and white images  
D) Requiring external API access for all operations

---

## Question 9
Vision Transformer (ViT) divides images into patches and treats them as sequences. What role does the learnable [cls] token play?

A) It marks the start of a new image in batch processing  
B) It stores spatial position information for the model  
C) Its final representation serves as the image embedding after transformer processing  
D) It computes similarity between patches

---

## Question 10
In multimodal retrieval at scale, approximate nearest neighbor search methods are preferred over exhaustive search primarily because:

A) They provide perfectly accurate results  
B) They are faster while maintaining acceptable accuracy, enabling practical large-scale deployment  
C) They eliminate the cold-start problem  
D) They guarantee symmetric matching between images and text

---

## Question 11
Video understanding models extend image understanding by incorporating temporal dynamics. Which approach uses motion between frames as an explicit signal?

A) 3D CNNs  
B) Temporal Transformers  
C) Optical Flow-Based Methods  
D) Frame Sampling Methods

---

## Question 12
Whisper (OpenAI's audio-language model) was trained on approximately how many hours of multilingual audio?

A) 68K hours  
B) 680K hours  
C) 6.8M hours  
D) 68M hours

---

## Question 13
For text-to-image evaluation, which metric measures realism by comparing feature distributions of generated and real images?

A) CLIP Score  
B) BLEU Score  
C) Fréchet Inception Distance (FID)  
D) CIDEr Score

---

## Question 14
Hallucination in vision-language models most directly arises from which of the following?

A) Using Vision Transformers instead of CNNs  
B) The language model component generating plausible descriptions independent of visual grounding, combined with weak vision encoders  
C) Insufficient training data  
D) Using only contrastive learning objectives

---

## Question 15
The modality gap refers to fundamental differences between modalities. Which statement best describes the relationship between image-to-text and text-to-image conversion?

A) Both are equally lossless transformations  
B) Image-to-text is lossless while text-to-image is ambiguous  
C) Image-to-text is lossy (details lost) while text-to-image is ambiguous (multiple valid images), creating asymmetric information flow  
D) Both conversions are perfectly symmetric and reversible

---

## Answer Key

**Question 1: B**
Early fusion concatenates raw or minimally-processed features directly, enabling deep modality interactions but requiring synchronized, aligned data. Late fusion processes modalities independently first (more modular), and hybrid fusion combines both approaches.

**Question 2: B**
In CLIP's contrastive training on N image-caption pairs, the matching pair is positive, and the N-1 other captions in the batch become negatives. The loss pushes mismatched image-caption combinations apart.

**Question 3: B**
CLIP's zero-shot capability works because the model learned to align images with diverse text descriptions, generalizing to new, unseen class labels. This is possible without any task-specific fine-tuning by leveraging the learned embedding space.

**Question 4: C**
The Q-Former (Query Transformer) in BLIP-2 uses learnable queries to bridge frozen image and text encoders, reducing training cost while maintaining performance. This lightweight adapter design is key to BLIP-2's efficiency.

**Question 5: B**
GPT-4V can understand and analyze images but cannot modify or generate new images. Its capabilities include VQA, document understanding, and scene understanding, but image generation/modification requires separate models like DALL-E.

**Question 6: B**
VQA models often exploit statistical biases in datasets (e.g., "sky" questions predominantly answered "blue") rather than achieving genuine visual understanding. This makes benchmark metrics misleading despite high reported accuracy.

**Question 7: C**
Self-critical training uses reinforcement learning to optimize caption-level metrics directly (BLEU, METEOR, CIDEr) instead of token-level cross-entropy loss. This better aligns training with evaluation metrics and improves caption quality.

**Question 8: B**
Stable Diffusion performs diffusion in a learned latent space (via autoencoder) rather than pixel space, drastically reducing inference time (~30 seconds vs. minutes). This makes it more practical for deployment while maintaining quality.

**Question 9: C**
The [cls] token is prepended to patch embeddings (similar to BERT), and its final representation after transformer processing serves as the overall image embedding. It aggregates spatial information across all patches.

**Question 10: B**
Approximate nearest neighbor methods (FAISS, Annoy) sacrifice some accuracy for speed, enabling practical deployment at scale. Exhaustive search is perfect but too slow for millions of embeddings, making the approximation worthwhile.

**Question 11: C**
Optical flow-based methods explicitly compute motion (differences between frame pairs) and use this as an additional signal. 3D CNNs and temporal transformers implicitly learn temporal patterns but don't explicitly compute flow.

**Question 12: B**
Whisper was trained on 680K hours of multilingual audio from the web, giving it exceptional robustness across languages, accents, and noisy conditions. This massive scale is key to its generalization capabilities.

**Question 13: C**
Fréchet Inception Distance (FID) measures the distance between feature distributions of generated and real images using a pre-trained classifier, providing a standard metric for image realism. CLIP Score measures alignment to text, not realism.

**Question 14: B**
Hallucination occurs when language model components generate plausible text without visual grounding, compounded by weak vision encoders that fail to constrain generation and training data containing descriptions of invisible elements.

**Question 15: C**
The modality gap creates asymmetric information loss: images carry rich details that can't be fully captured in text (image→text is lossy), while text descriptions are often ambiguous with multiple valid images matching the same description (text→image is ambiguous).

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
