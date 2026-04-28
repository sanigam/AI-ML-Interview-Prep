# Multiple Choice Questions: Computer Vision — Object Detection and Segmentation

📺 **Video Lecture:** https://youtu.be/gMu8VeBheUc


## Question 1
In object detection, why is handling variable numbers of objects more challenging than image classification?

A) Image classification only requires detecting one object while detection requires detecting many  
B) Objects in detection datasets are always smaller than in classification datasets  
C) Detection must produce variable-length outputs (0 to N objects) with both class labels and bounding boxes, unlike classification's fixed single output  
D) Classification networks cannot be reused for detection tasks

---

## Question 2
What is the primary advantage of anchor-free detection methods (YOLO v1, CenterNet, FCOS) compared to anchor-based approaches?

A) They eliminate the need for post-processing like Non-Maximum Suppression  
B) They require less computational resources than anchor-based methods regardless of model size  
C) They achieve higher accuracy on all dataset types without exception  
D) They eliminate hyperparameter tuning for anchor sizes and generate fewer default boxes during inference

---

## Question 3
In the R-CNN family of two-stage detectors, what is the purpose of RoI pooling?

A) To apply non-maximum suppression to overlapping regions  
B) To extract fixed-size feature representations from variable-sized region proposals, enabling efficient parallel classification  
C) To generate region proposals from the backbone features  
D) To remove duplicate detections using intersection over union

---

## Question 4
How does YOLO v2/v3 improve upon the original YOLO v1 to better detect small objects?

A) By removing anchor boxes entirely from the prediction pipeline  
B) By using larger input image sizes exclusively  
C) By adding multi-scale predictions (detecting small objects on high-resolution feature maps, large objects on low-resolution)  
D) By abandoning convolutional networks in favor of fully connected layers

---

## Question 5
Focal loss is particularly important for one-stage detectors. What problem does it solve?

A) Class imbalance where >99% of predicted regions are background, causing the model to be dominated by easy negatives  
B) The inability to predict variable numbers of objects per image  
C) The problem of detecting objects at multiple scales in an image  
D) The computational inefficiency of processing multiple anchor boxes

---

## Question 6
In the context of bounding box post-processing, how does Soft-NMS improve upon standard NMS?

A) It decays confidence scores of overlapping boxes instead of hard-removing them, preserving nearby detections of distinct objects  
B) It uses IoU calculated with respect to enclosing area instead of union area  
C) It sorts boxes by IoU threshold instead of by confidence score  
D) It completely removes overlapping boxes instead of just suppressing them

---

## Question 7
What is the main limitation of standard Intersection over Union (IoU) as a loss function for bounding box regression?

A) It requires pre-defined anchor boxes to be meaningful  
B) It only works for square bounding boxes, not rectangular ones  
C) It cannot distinguish between anchor-based and anchor-free detectors  
D) For non-overlapping boxes, IoU=0 provides zero gradient, preventing the model from learning to move the box toward the target

---

## Question 8
Why are Feature Pyramid Networks (FPN) essential for modern object detectors?

A) They remove the need for post-processing like Non-Maximum Suppression  
B) They eliminate the need for data augmentation during training  
C) They create a multi-scale feature representation where predictions at each scale have high semantic content, enabling effective detection across object sizes  
D) They guarantee that two-stage detectors outperform one-stage detectors

---

## Question 9
What is the primary difference between semantic segmentation and instance segmentation?

A) Semantic segmentation uses CNNs while instance segmentation uses transformers  
B) Instance segmentation is always faster than semantic segmentation  
C) Semantic segmentation requires more training data than instance segmentation  
D) Semantic segmentation assigns class labels to each pixel but cannot distinguish different instances of the same class; instance segmentation produces separate masks for each distinct object

---

## Question 10
How does Mask R-CNN extend Faster R-CNN to perform instance segmentation?

A) By adding a parallel mask prediction branch that outputs a binary segmentation mask for each region proposal using RoI Align  
B) By training two separate models: one for detection and one for segmentation  
C) By removing the RPN and predicting masks directly on the backbone features  
D) By replacing the bounding box regression branch with a mask prediction branch

---

## Question 11
What is the advantage of using RoI Align over RoI Pooling in Mask R-CNN?

A) RoI Align uses bilinear interpolation to align features to region boundaries precisely (differentiable), improving mask accuracy compared to RoI Pooling's coordinate quantization  
B) RoI Align allows for anchor-free detection unlike RoI Pooling  
C) RoI Align eliminates the need for the RPN  
D) RoI Align is faster than RoI Pooling

---

## Question 12
In the context of real-time object detection optimization for a latency budget of 33ms (30 FPS), which strategy would be most effective?

A) Always use the largest available backbone model (ResNet-152) for maximum accuracy  
B) Disable all data augmentation to speed up training  
C) Process images sequentially on CPU to avoid GPU overhead  
D) Use efficient backbones (MobileNet, ShuffleNet), quantize to int8, and apply knowledge distillation while accepting a small accuracy drop

---

## Question 13
What distinguishes DETR (Detection Transformer) from traditional CNN-based detectors in its fundamental approach?

A) It is significantly slower than all CNN-based detectors without exception  
B) It is the first detector to use convolutional neural networks  
C) It reformulates detection as a sequence prediction problem using transformer encoder-decoders with object queries, eliminating the need for NMS post-processing  
D) It can only detect one object per image

---

## Question 14
Mosaic data augmentation is particularly beneficial for object detection. Why is it more effective than standard augmentation (crops, flips, brightness changes)?

A) It only works with anchor-free detectors like YOLO v1  
B) It guarantees perfect accuracy without any false negatives  
C) It creates augmented images from 4 random images, forcing the model to detect objects at different scales and positions within one image, improving small object detection and increasing effective batch diversity  
D) It completely eliminates the need for bounding box annotations

---

## Question 15
When designing an end-to-end autonomous driving detection system with a <100ms latency budget, which architectural choice and trade-off is most appropriate?

A) Use the highest-accuracy two-stage detector (Cascade R-CNN) regardless of latency, as accuracy is paramount  
B) Use semantic segmentation instead of object detection for faster processing  
C) Use an optimized one-stage detector (quantized YOLOv8) with acceptable mAP (~40) to meet real-time constraints, accepting some accuracy loss for operational feasibility  
D) Use an anchor-based detector exclusively, as anchor-free methods are always slower

---

## Answer Key

**Q1: C**
Detection requires variable-length outputs because multiple objects can appear in an image, each needing a class label and bounding box. Classification produces a single prediction per image. This fundamentally complicates the task compared to pure classification.

**Q2: D**
Anchor-free methods eliminate the need to manually design anchor box sizes and aspect ratios, reducing hyperparameter tuning overhead. They generate fewer default boxes and can handle irregular aspect ratios better. DETR still requires post-processing in some variants, so A is not correct.

**Q3: B**
RoI pooling crops features corresponding to region proposals and applies max pooling to produce fixed-size feature maps, enabling efficient parallel processing of all proposals. This is critical for Faster R-CNN's speed advantage over R-CNN.

**Q4: C**
YOLO v2/v3 introduced multi-scale predictions—detecting small objects on high-resolution feature maps (which preserve fine details) and large objects on low-resolution feature maps (which have broader context). This directly addresses YOLO v1's weakness with small objects.

**Q5: A**
The class imbalance problem (>99% background in one-stage detectors) causes standard loss to be dominated by easy negatives. Focal loss down-weights easy examples and focuses on hard positives, enabling one-stage detectors to match two-stage accuracy.

**Q6: A**
Soft-NMS decays confidence scores of overlapping boxes rather than hard-removing them, allowing nearby detections to be preserved with reduced confidence. This helps when legitimate objects overlap slightly or are in close proximity.

**Q7: D**
For non-overlapping predicted and ground-truth boxes, IoU=0 (no intersection, no union overlap), providing zero gradient signal. This prevents backpropagation from learning to move the box toward the target. GIoU and related metrics solve this.

**Q8: C**
FPN combines high-resolution low-semantic features from early layers with low-resolution high-semantic features from deep layers. This multi-scale representation ensures that predictions at every scale have strong semantic content, critical for detecting objects ranging from tiny to large.

**Q9: D**
Semantic segmentation outputs one class label per pixel but treats all instances of the same class as one region. Instance segmentation separately identifies and masks each individual object, distinguishing between two dogs as two separate masks.

**Q10: A**
Mask R-CNN adds a mask prediction branch in parallel to the existing classification and bounding box branches. Each region proposal generates not just a class and refined box, but also a binary mask (28x28) indicating pixel membership in the object.

**Q11: A**
RoI Align uses bilinear interpolation to align features to region boundaries precisely, making the operation differentiable and improving mask quality. RoI Pooling quantizes region coordinates to grid points, losing precision in feature alignment.

**Q12: D**
With 33ms budget, efficient backbones (MobileNet ~10-20ms inference), int8 quantization (2-4x speedup), and knowledge distillation maintain acceptable accuracy. Large models would exceed the latency budget; disabling augmentation harms accuracy.

**Q13: C**
DETR uses transformer decoder with learned object queries, outputting all objects in parallel without NMS. Traditional detectors predict grids of boxes; DETR directly predicts N object predictions, fundamentally changing the detection paradigm.

**Q14: C**
Mosaic combines 4 random images into a 2x2 grid, creating diverse object scales and positions within a single image. This increases effective batch diversity and exposes the model to small objects in varied positions, improving small object detection by ~5% mAP.

**Q15: C**
Autonomous driving requires <100ms latency; one-stage detectors with optimization (quantization) are necessary. Accepting ~40 mAP instead of pursuing maximum accuracy (which would require slower two-stage detectors) is the practical trade-off for real-time operation.

---

*© 2026 AI Nirvana · More Info: https://medium.com/@snigam/a-simple-structured-way-to-prepare-for-ai-ml-interviews-68b2e5830195 · Disclaimer: Provided as is. No liability assumed.*
