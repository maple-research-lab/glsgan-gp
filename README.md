# Loss-Sensitive GAN

Framework: Pytorch 0.2.0.post1 or above

Language: Python 2.7 or above

Implemented by: Zihang Zou, [[Laboratory for MAchine Perception and Learning(MAPLE)](http://maple.cs.ucf.edu)], University of Central Florida

Please cite the following paper when referring to the following algorithms:

**Guo-Jn Qi. Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities. arXiv:1701.06264 [[pdf](https://arxiv.org/abs/1701.06264)]**

## Reference

### LS-GAN

Code: https://github.com/guojunq/lsgan

Paper: https://arxiv.org/pdf/1701.06264

This implementation covers the algorithms proposed in loss-sensitive GAN, including direct gradient penalty and generalized loss sensitive GAN. The loss sensitive gan regularizes GAN on Lipschitz density through a margin and well defined discrminator output. LSGAN abandons the the binary entropy term proposed in original GAN and apply the assumption that a real example should have a smaller loss than a generated sample. Discrmintor loss and generator loss for GLS-GAN are as below:

D_loss = LeakyReLU(D(x) - D(G(z)) + lambda * \delta(x, G(z))).mean()

G_loss = D(G(z)).mean()

It's worthy noted that we use LeakyReLU for generalized LS-GAN. And this function is only a special case of ()+ from the original proof. Any other ()+ function also works under the generalized theorem of LS-GAN.

The gradient penalty applies the form proposed originally in the first version of LS-GAN, Chapter 5 [[pdf](https://arxiv.org/pdf/1701.06264v1.pdf)], quoted here
"Alternatively, one may consider to directly minimize
the gradient norm ||∇xLθ(x)|| as a regularizer for
the LS-GAN. In this paper, we adopt weight decay for its
simplicity and find it works well with the LS-GAN model
in experiments."

## Usage
### 1.PYTORCH version
1.In this implementation, we use the following version of PYTORCH (any version beyond this will also work), 
``` bash
$ pip list | grep torch
torch (0.2.0.post1)
torchvision (0.1.8)
```
We use the following function to calculate the gradient penalty.
torch.autograd.grad() [[source](https://github.com/pytorch/pytorch/blob/master/torch/autograd/__init__.py)]

### 2.Download dataset
1.Setup and download celebA dataset 

Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".


2.Crop the face using face detector.
``` bash
$ python ./Data/face_detect.py
```
Note: For those dataset that are not supported by PYTORCH, you can use your own image folder by using the parameter --dataset folder, the code will work. And be sure to have a sub-folder under the main images folder. For example, celebA_crop/64_crop/.

### 3.Train LS-GAN
The default slope is 0, which is LS-GAN,
```bash
$ python lsgan-gp.py --dataset folder --dataroot celebA_crop --cuda --niter 25
```
If slope is set to 1, it is WGAN,
```bash
$ python lsgan-gp.py --dataset folder --dataroot celebA_crop --cuda --niter 25 --slope 1
```
Or you can explore more slope as GLS-GAN, for example,
```bash
$ python lsgan-gp.py --dataset folder --dataroot celebA_crop --cuda --niter 25 --slope 0.01
```

## Results
We save our generated images in samples folder using torchvision.utils.save_image function.
You should get the following results after running the code.

LSGAN converges faster! You can start getting recognizable results after half an epoch.

### half epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_half_epoch.jpg)

### 1 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_1_epoch.jpg)

### 2 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_2_epoch.jpg)

### 3 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_3_epoch.jpg)

### 5 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_5_epoch.jpg)

### 10 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_10_epoch.jpg)

### 15 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_15_epoch.jpg)

### 20 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_20_epoch.jpg)

### 25 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/crop_lsgan_gp_25_epoch.jpg)

