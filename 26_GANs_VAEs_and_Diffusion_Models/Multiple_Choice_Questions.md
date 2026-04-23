# Multiple Choice Questions: GANs, VAEs, and Diffusion Models

Test your understanding of generative model concepts for AI/ML interviews.

---

**Q1. In a GAN, the generator and discriminator are trained with:**

A) The same loss function
B) Opposing objectives — the generator tries to fool the discriminator while the discriminator tries to distinguish real from fake
C) Only unsupervised losses
D) Reinforcement learning rewards

---

**Q2. Mode collapse in GANs occurs when:**

A) The discriminator becomes too weak
B) The generator produces only a limited subset of the data distribution, ignoring other modes
C) Training converges to the global optimum
D) The learning rate is too small

---

**Q3. The Wasserstein GAN (WGAN) addresses GAN training instability by:**

A) Using a larger generator
B) Replacing the binary classification loss with the Wasserstein distance, providing continuous gradients even when the discriminator is confident
C) Removing the discriminator entirely
D) Training only the generator

---

**Q4. In a Variational Autoencoder (VAE), the loss function (ELBO) consists of:**

A) Only reconstruction loss
B) Reconstruction loss + KL divergence between the learned latent distribution and a prior (typically N(0,I))
C) Only KL divergence
D) Cross-entropy loss on class labels

---

**Q5. The reparameterization trick in VAEs enables backpropagation through sampling by:**

A) Removing the sampling step entirely
B) Expressing z = μ + σ·ε where ε ~ N(0,I), making gradients flow through μ and σ
C) Using a discrete distribution instead
D) Applying dropout to the latent space

---

**Q6. Compared to GANs, VAEs typically produce:**

A) Sharper but less diverse images
B) Blurrier images but with a more structured, interpolable latent space
C) Identical quality images
D) No images at all

---

**Q7. Conditional GANs (cGANs) differ from standard GANs by:**

A) Not using a discriminator
B) Conditioning both generator and discriminator on auxiliary information (e.g., class labels or text)
C) Only generating random noise
D) Using reinforcement learning

---

**Q8. Diffusion models generate samples by:**

A) A single forward pass through a generator network
B) Gradually denoising random noise through many iterative steps, reversing a forward noise-adding process
C) Adversarial training between two networks
D) Maximizing a discriminator's confidence

---

**Q9. The forward process in a diffusion model:**

A) Generates high-quality images directly
B) Gradually adds Gaussian noise to data over T steps until it becomes pure noise
C) Trains the discriminator
D) Compresses images into a latent space

---

**Q10. A key advantage of diffusion models over GANs is:**

A) Faster sampling speed
B) More stable training and better mode coverage (less mode collapse)
C) Fewer parameters required
D) No need for neural networks

---

**Q11. Latent diffusion models (used in Stable Diffusion) improve efficiency by:**

A) Running the diffusion process in pixel space
B) Running the diffusion process in a compressed latent space (via a VAE encoder), reducing computational cost
C) Eliminating the denoising steps
D) Using only text input without any image data

---

**Q12. The KL divergence term in the VAE loss encourages the latent distribution to:**

A) Maximize reconstruction quality
B) Match the prior distribution N(0,I), ensuring the latent space is smooth and enables meaningful sampling
C) Increase to infinity
D) Be identical to the output distribution

---

**Q13. DALL-E and Stable Diffusion generate images from text by:**

A) Using GANs exclusively
B) Conditioning diffusion models on text embeddings (e.g., from CLIP) to guide the denoising process
C) Searching a database of existing images
D) Using template matching

---

**Q14. Spectral normalization in GANs is used to:**

A) Speed up the generator
B) Stabilize discriminator training by constraining the Lipschitz constant of its weight matrices
C) Increase mode collapse
D) Remove the need for a loss function

---

**Q15. The main disadvantage of diffusion models compared to GANs is:**

A) Lower image quality
B) Slow sampling due to many iterative denoising steps (often 20-1000 steps)
C) Inability to generate high-resolution images
D) Requirement for paired training data

---

## Answer Key

**Q1. Answer: B**
GANs use an adversarial (minimax) objective: the generator minimizes the probability of the discriminator correctly classifying its outputs as fake, while the discriminator maximizes classification accuracy.

**Q2. Answer: B**
Mode collapse means the generator finds a few outputs that fool the discriminator and repeatedly produces only those, ignoring the full diversity of the real data distribution.

**Q3. Answer: B**
WGAN uses the Earth Mover's (Wasserstein) distance instead of JS divergence. This provides meaningful gradients even when distributions don't overlap, significantly stabilizing training.

**Q4. Answer: B**
The ELBO = E[log p(x|z)] − KL(q(z|x) || p(z)). Reconstruction ensures faithful outputs; KL regularization ensures the latent space is structured and enables sampling from the prior.

**Q5. Answer: B**
Instead of sampling z ~ N(μ, σ²) directly (non-differentiable), we sample ε ~ N(0,I) and compute z = μ + σ·ε. Gradients now flow through μ and σ while ε is treated as a constant.

**Q6. Answer: B**
VAEs produce blurrier outputs due to the KL regularization (preventing the latent space from encoding too much detail), but offer smooth interpolation and stable training. GANs produce sharper but potentially less diverse images.

**Q7. Answer: B**
cGANs provide conditioning information (class label, text description) to both G and D. G generates conditioned on this input, and D judges whether the (sample, condition) pair is realistic.

**Q8. Answer: B**
Diffusion models learn to reverse a gradual noising process. Starting from pure noise, the model iteratively predicts and removes noise at each step, progressively constructing a clean sample.

**Q9. Answer: B**
The forward (noising) process adds small amounts of Gaussian noise at each of T timesteps, gradually transforming structured data into isotropic Gaussian noise. The reverse process learns to undo this.

**Q10. Answer: B**
Diffusion models don't use adversarial training, avoiding mode collapse and training instability. They optimize a simple denoising objective, providing more stable training and better coverage of the data distribution.

**Q11. Answer: B**
Latent diffusion first encodes images to a lower-dimensional latent space using a VAE, then runs the diffusion process there. This dramatically reduces computation while maintaining generation quality.

**Q12. Answer: B**
The KL term regularizes the encoder to produce distributions close to N(0,I). This ensures the latent space has no "holes" — any point sampled from N(0,I) maps to a reasonable output.

**Q13. Answer: B**
Text-to-image models use text encoders (often CLIP) to create conditioning embeddings that guide the diffusion denoising process, steering generation toward images matching the text description.

**Q14. Answer: B**
Spectral normalization divides weight matrices by their spectral norm (largest singular value), constraining the discriminator's Lipschitz constant and preventing it from becoming too powerful too quickly.

**Q15. Answer: B**
Diffusion models require many sequential denoising steps at inference time (typically 20-50 with DDIM, up to 1000 with DDPM), making generation much slower than GANs' single forward pass.

---

*© 2026 AI Nirvana · Disclaimer: Provided as is. No liability assumed.*
