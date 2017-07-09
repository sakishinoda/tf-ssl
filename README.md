# tf-ssl: TensorFlow Semi-Supervised Library

Implementations of the state-of-the-art for semi-supervised deep learning.

Performance benchmarks on MNIST, CIFAR-10, CIFAR-100, SVHN below (and [here](http://www.sakishinoda.com/2017/07/05/semisupervised-deep-learning-classification-results.html)). 

---


#### Permutation Invariant MNIST (Average Error Rate, %)

| 100 labels  | 1000 labels   | All labels | Method   | Year  |
|---|---|---|---|---| 
| 0.93 (±0.065) | N/A | N/A | [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) | 2016 |
| 1.002 (±0.038)| 0.979 (±0.025) | 0.578 (±0.013) | [Deconstructing the Ladder Network Architecture](http://arxiv.org/abs/1511.06430) (Ladder w/ AMLP[2,2,2]) | 2015 |
| 1.072 (±0.015)| 0.974 (±0.021) | 0.598 (±0.014) | [Deconstructing the Ladder Network Architecture](http://arxiv.org/abs/1511.06430) (Ladder w/ AMLP[4]) | 2015 |
| 1.072 (±0.015)| 1.193 (±0.039) | 0.569 (±0.010) | [Deconstructing the Ladder Network Architecture](http://arxiv.org/abs/1511.06430) (Ladder w/ AMLP[2,2]) | 2015 |
| 1.06 (±0.37) | 0.84 (±0.08) | 0.57 (±0.02) | [Semi-Supervised Learning with Ladder Networks](http://arxiv.org/abs/1507.02672) | 2015 |
| 1.36 | 1.27 | 0.64 | [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](http://arxiv.org/abs/1704.03976) | 2017 |
| 2.33 | 1.36 | 0.637 (±0.046) | [Distributional Smoothing with Virtual Adversarial Training](https://arxiv.org/abs/1507.00677) | 2016 (ICLR) |



#### CIFAR-10 (Average Error Rate, %)

| 4k labels | All labels | Method | Year |
|---|---|---|---|
| 10.55 | N/A | [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](http://arxiv.org/abs/1704.03976) [Conv-Large w/ EntMin, w/ augmentation] | 2017 |
| 12.16 (±0.24)| 5.60 (±0.10) | [Temporal Ensembling for Semi-Supervised Learning](http://arxiv.org/abs/1610.02242) [w/ augmentation]  | 2016 |
| 13.15 | N/A | [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](http://arxiv.org/abs/1704.03976) [Conv-Large w/ EntMin, no augmentation] | 2017 |
| 20.40 | N/A | [Semi-Supervised Learning with Ladder Networks](http://arxiv.org/abs/1507.02672) [Conv-Large, Gamma model, no augmentation] | 2015 |


#### SVHN (Average Error Rate, %)

| 500 labels | 1000 labels | All labels | Method | Year |
|---|---|---|---|---|
| N/A | 3.86 | N/A | [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](http://arxiv.org/abs/1704.03976) [Conv-Large w/ EntMin, w/ augmentation] | 2017 |
| N/A | 4.28 | N/A | [Virtual Adversarial Training: a Regularization Method for Supervised and Semi-supervised Learning](http://arxiv.org/abs/1704.03976) [Conv-Large w/ EntMin, no augmentation] | 2017 |
| 5.12 (±0.13) | 4.42 (±0.16) | 2.74 (±0.06) | [Temporal Ensembling for Semi-Supervised Learning](http://arxiv.org/abs/1610.02242) [w/ augmentation] | 2016 |
| 6.65 (±0.53) | 4.82 (±0.17) | 2.54 (±0.04) | [Temporal Ensembling for Semi-Supervised Learning](http://arxiv.org/abs/1610.02242) [Pi model w/ augmentation] | 2016 |
| N/A | 24.63 | N/A | [Distributional Smoothing with Virtual Adversarial Training](https://arxiv.org/abs/1507.00677) | 2016 (ICLR) | 



#### CIFAR-100 (Average Error Rate, %)
| 10k labels | All labels | Random 500k Tiny Images | Restricted 237k Tiny Images | Method | Year |
|---|---|---|---|---|--|
| 38.65 (±0.51) | 26.30 (±0.15) | 23.62 (±0.23) | 23.79 (±0.24) | [Temporal Ensembling for Semi-Supervised Learning](http://arxiv.org/abs/1610.02242) | 2016 |
