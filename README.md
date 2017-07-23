# Loss-Sensitive GAN With Gradient Penalty

Implemented by: Zihang Zou, Laboratory for MAchine Perception and Learning(MAPLE), University of Central Florida

Please cite the following paper when referring to the following algorithms:

**Guo-Jn Qi. Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities. arXiv:1701.06264 [[pdf](https://arxiv.org/abs/1701.06264)]**

## Reference

### 1.LS-GAN

Code: https://github.com/guojunq/lsgan

Paper: https://arxiv.org/pdf/1701.06264

About gradient penalty of LS-GAN, Dr.Qi proposed it in the first version of LS-GAN, Chapter 5 [[pdf](https://arxiv.org/pdf/1701.06264v1.pdf)]
"Alternatively, one may consider to directly minimize
the gradient norm ||∇xLθ(x)|| as a regularizer for
the LS-GAN. In this paper, we adopt weight decay for its
simplicity and find it works well with the LS-GAN model
in experiments."

### 2.WGAN

Code: https://github.com/martinarjovsky/WassersteinGAN

Paper: https://arxiv.org/pdf/1701.07875

### 3.WGAN-GP

Code: https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L129

Paper: https://arxiv.org/pdf/1704.00028.pdf#Chapter4

I referenced the above paper and code to implement the gradient penalty.

Note: I used '#' in the above links to let you know the detail locations. Since those link are pdf files instead of html it will not jump to the anchor automatically.

## Usage
### 1.PYTORCH version
1.In this implementation, we use the following version of PYTORCH, 
``` bash
$ pip list | grep torch
torch (0.1.12.post2)
torchvision (0.1.8)
```
I estimate the gradient penalty by 2 random samples, please check the code for more details.

However, if you want to calculate the true gradient penalty, you need to use the master version of PYTORCH which has the function: torch.autograd.grad() [[source](https://github.com/pytorch/pytorch/blob/master/torch/autograd/__init__.py)]

Then you will be challenged with compiling PYTORCH on your local server.

### 2.Download dataset
1.Setup and download celebA dataset 

Download img_align_celeba.zip from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the link "Align&Cropped Images".

Note: For those dataset that are not supported by PYTORCH, you can create your own image folder and use the --dataset folder, the code will work. And be sure to have a sub-folder under the main images folder. For example, celebA/img_align.

### 3.Train LS-GAN-GP
```bash
$ python lsgan-gp.py --dataset folder --dataroot celebA --cuda --niter 25
```

## Results
We save our generated images in samples folder using torchvision.utils.save_image function.
You should get the following results after running the code.

### 1 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/1_epoch.jpg)

### 3 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/3_epoch.jpg)

### 5 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/5_epoch.jpg)

### 12 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/12_epoch.jpg)

### 24 epoch
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/24_epoch.jpg)

### More results in 24 epoch

![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/fake_samples_24_00002900.jpg)
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/fake_samples_24_00003000.jpg)
![alt text](https://github.com/zzzucf/lsgan-gp/blob/master/results/fake_samples_24_00003100.jpg)
