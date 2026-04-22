# Multiple Choice Questions: Convolutional Neural Networks

📺 **Video Lecture:** https://youtu.be/K1xfB-H_Ii4


Test your understanding of CNN concepts essential for AI/ML interviews.

---

**Q1. The output spatial dimension of a convolution with input size I, filter size F, padding P, and stride S is:**

A) (I + F − 2P) / S + 1
B) (I − F + 2P) / S + 1
C) (I × F + P) / S
D) I − F + P + S

---

**Q2. The key advantage of convolution over fully connected layers for image data is:**

A) Convolution uses more parameters
B) Parameter sharing and local connectivity exploit spatial structure
C) Convolution eliminates the need for activation functions
D) Convolution always produces higher accuracy

---

**Q3. Max pooling with a 2×2 window and stride 2:**

A) Doubles the spatial dimensions
B) Halves each spatial dimension while retaining the most salient features
C) Reduces the number of channels by half
D) Applies a learnable transformation to the input

---

**Q4. ResNet's skip connection computes y = F(x) + x. If the network needs to learn an identity mapping, F(x) must learn:**

A) The input x
B) Zero (the residual is zero)
C) A random transformation
D) The softmax of x

---

**Q5. A 1×1 convolution is primarily used for:**

A) Spatial feature extraction across neighboring pixels
B) Channel-wise dimensionality reduction or expansion
C) Replacing pooling layers
D) Increasing the receptive field

---

**Q6. The receptive field of a neuron in a deep CNN refers to:**

A) The number of parameters in the filter
B) The region of the original input that influences that neuron's activation
C) The output size of the final layer
D) The learning rate used during training

---

**Q7. Depthwise separable convolutions reduce computation compared to standard convolutions by:**

A) Using larger filters
B) Separating spatial filtering (depthwise) from channel mixing (pointwise 1×1)
C) Eliminating the bias term
D) Using fewer layers in the network

---

**Q8. Transfer learning with a pre-trained ImageNet model is effective because:**

A) ImageNet models memorize all possible images
B) Early layers learn general visual features (edges, textures) that transfer across tasks
C) ImageNet models never overfit
D) Pre-trained weights cannot be modified

---

**Q9. Which data augmentation technique is NOT label-preserving for standard image classification?**

A) Horizontal flip of a car image
B) Small rotation (±15 degrees)
C) Flipping a medical X-ray left-to-right (where laterality matters)
D) Random brightness adjustment

---

**Q10. In object detection, Faster R-CNN is classified as a:**

A) Single-shot detector
B) Two-stage detector (region proposals then classification)
C) Anchor-free detector
D) Unsupervised detector

---

**Q11. AlexNet (2012) was a breakthrough primarily because it:**

A) Introduced the convolution operation
B) Demonstrated that deep CNNs trained on GPUs could dramatically outperform hand-crafted features on ImageNet
C) Was the first neural network ever built
D) Used only 1×1 convolutions

---

**Q12. In semantic segmentation, the encoder-decoder architecture is needed because:**

A) Classification requires spatial resolution
B) The encoder downsamples for context and the decoder upsamples to restore pixel-level predictions
C) It eliminates the need for training data
D) It replaces the loss function

---

**Q13. Vision Transformer (ViT) processes images by:**

A) Applying convolutions followed by pooling
B) Dividing the image into patches and treating them as tokens for a transformer encoder
C) Using recurrent connections across pixels
D) Applying batch normalization to raw pixels

---

**Q14. Knowledge distillation trains a small student network by:**

A) Copying the teacher's weights directly
B) Matching the student's output distribution to the teacher's soft probability outputs
C) Training only on hard labels
D) Increasing the student's depth to match the teacher

---

**Q15. VGG networks demonstrated the importance of:**

A) Using very large convolutional filters (11×11)
B) Network depth with small (3×3) filters stacked repeatedly
C) Skip connections between layers
D) Depthwise separable convolutions

---

## Answer Key

**Q1. Answer: B**
The standard formula is Output = (I − F + 2P) / S + 1. For example, a 28×28 input with a 3×3 filter, padding 1, stride 1 gives (28 − 3 + 2) / 1 + 1 = 28.

**Q2. Answer: B**
Convolution shares a single set of filter weights across all spatial positions (parameter sharing) and connects each output to only a local input region (local connectivity), dramatically reducing parameters compared to fully connected layers.

**Q3. Answer: B**
A 2×2 max pooling with stride 2 takes the maximum value from each non-overlapping 2×2 region, halving the height and width while retaining the most activated feature in each region.

**Q4. Answer: B**
If identity is optimal, F(x) just needs to output zeros, and the skip connection passes x through unchanged. Learning "do nothing" (zero residual) is easier than learning to replicate the input.

**Q5. Answer: B**
A 1×1 convolution applies a linear transformation across channels at each spatial position, enabling cheap dimensionality reduction (fewer channels) or expansion without spatial mixing.

**Q6. Answer: B**
The receptive field is the area of the original input image that contributes to a particular neuron's value. It grows with network depth, stride, and filter size.

**Q7. Answer: B**
Depthwise separable convolutions first apply a separate spatial filter per channel (depthwise), then mix channels via 1×1 convolutions (pointwise), reducing computation by roughly K² times for K×K filters.

**Q8. Answer: B**
Early CNN layers learn generic features like edges and textures that are useful across many visual tasks. Fine-tuning these pre-trained features on a new task requires far less data than training from scratch.

**Q9. Answer: C**
Flipping medical X-rays changes the laterality (left vs. right), which may be diagnostically significant. This violates label preservation. Standard augmentations must be chosen based on domain knowledge.

**Q10. Answer: B**
Faster R-CNN uses a Region Proposal Network (RPN) to generate candidate bounding boxes (stage 1), then classifies and refines each proposal (stage 2). It is more accurate but slower than single-shot detectors.

**Q11. Answer: B**
AlexNet won the 2012 ImageNet competition by a large margin, showing that deep CNNs with ReLU, dropout, and GPU training could surpass traditional computer vision methods, sparking the deep learning revolution.

**Q12. Answer: B**
The encoder reduces spatial resolution to build rich feature representations with large receptive fields. The decoder upsamples these features back to the original resolution for pixel-level classification.

**Q13. Answer: B**
ViT splits images into fixed-size patches (e.g., 16×16), linearly embeds each patch, adds position embeddings, and processes the sequence of patch tokens through a standard transformer encoder.

**Q14. Answer: B**
The student learns from the teacher's soft probability outputs (softened with temperature T), which contain richer information about class similarities than hard one-hot labels alone.

**Q15. Answer: B**
VGG showed that stacking many 3×3 convolutional layers achieves the same receptive field as larger filters (e.g., two 3×3 layers = one 5×5) with fewer parameters and more nonlinearity.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
