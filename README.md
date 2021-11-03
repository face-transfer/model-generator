# StyleGAN3
NVidia's StyleGAN series of papers and models aim at generating synthetic high resolution images from real images with controled modifications.
It provides a ready to use set of networks to generate fake images and a curated dataset of high quality images (FFHQ for Flickr Faces High Quality).
Applications are in photo edition (photoshop like), domain translation (winter to summer, old to young, etc.) and video generation.
It is AI and ANN powered. The NN learns from a set of images (either human faces, or animal faces, or cars, etc) over several days on multiple high-end GPUs. 
It can as well be used as a pre-trained model and not require all the training and either fine-tuning it through transfer learning or just using the pre-trained models as is.


GAN
---
How this is achieved is all based on Generative Adversarial Networks --> GAN. Relying on game theory, this architecture enables to put in competition 2 players, each of which is a machine learning model. 1 model is a discriminator: it receives an image as input and its output is whether the input image is fake or real. The objective of this model is to be as accurate as possible in its discrimination of real and fake images.

The other player model is a generator. It is just specified the characteristics of the desired output and it yields this output. Its objective is to produce an image as real-looking as possible and thus deceiving the discriminator.

The competition can be formalized in technical terms by a minimax game with a value function reflecting the binary cross-entropy and the sochasticity of the process.


GAN -> StyleGAN
---------------
On top of this GAN architecture, StyleGAN enabled to build an image as a hierarchical synthesis of various layers of details of an image. The point is to control the various aspects of the desired output image at various resolution levels. It was enabled by 

. disentanglement. Disentanglement was obtained among others through progressively growing GANs with finer resolution as we go deeper in the overall network (from coarse features like pose, and overall color, to face shape, to hair, or even stubble and freckles) 

. start generation from a constant and not a latent variable. Random noise is introduced in StyleGAN* at other levels. 

. integrate the learning into a mapping network and performing affine scaling and biasing of Adaptive Instance normalization on top at various levels and in each layer, then mixing the styles thus created; this replaces the pixelnorm original GAN normalization and enables control of styles for each feature

The quality of the result achieved is then measured through Frechet Inception Distance (measures distance between two densities) and Precision and Recall. 


StyleGAN -> StyleGAN2
---------------------
The next model solved 2 issues observed in the first version: droplet or blob effect and unhealthy constants of some features. Both are solved by

. keeping network topology fix and changing input image resolution at training stage instead to hold layered control of features

. taking new quality metrics: Perceptual Path Length, PPL, records small results to small changes in inputs as expected. Previously used Precision and Recall metric was blind to image quality. Together with normalizing lazily in turn, after separating normalization and modulation. This enables smoother interpolations and eliminates droplet artifact which resulted probably from disconnection between feature mapping and AdaIN instance normalization

. faster training


StyleGAN2 -> StyleGAN3
----------------------
Improvements include:

. resolving of alias artifact due to sampling mismatch with initial resolution

. texture sticking

. translation and rotation equivariance: in StyleGAN2, distortion of faces when being shifted or rotated; here solved in 2 configurations of the network: StyleGAN3-T and StyleGAN3-R (which covers both equivariances)


StyleGAN2-ADA for smaller input sets, with fewer images for training


# How to train NVidia's StyleGAN3 
PyTorch implementation of the NeurIPS 2021 paper
Different configurations for different input resolutions
Quoting Nvidia's documents,

Requirements

. Linux and Windows are supported, but the authors recommend Linux for performance and compatibility reasons.

. 1–8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using Tesla V100 and A100 GPUs.

. 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.

. CUDA toolkit 11.1 or later. (Why is a separate CUDA toolkit installation required? See Troubleshooting).

. GCC 7 or later (Linux) or Visual Studio (Windows) compilers. Recommended GCC version depends on CUDA version, see for example CUDA 11.4 system requirements.

. Python libraries: see environment.yml for exact library dependencies. You can use the following commands with Miniconda3 to create and activate your StyleGAN3 
. Python environment:
conda env create -f environment.yml
conda activate stylegan3

. Docker users:
Ensure you have correctly installed the NVIDIA container runtime.
Use the provided Dockerfile to build an image with the required library dependencies.
The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing Visual Studio Community Edition and adding it into PATH using "C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat".

See Troubleshooting for help on common installation and run-time problems.


Then, run for instance:

\# Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.

python train.py --outdir=\~/training-runs --cfg=stylegan3-r --data=\~/datasets/metfacesu-1024x1024.zip \
    --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
    --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl


# How to fine-train StyleGAN3? Transfer Learning
Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file dataset.json for labels. Custom datasets can be created from a folder containing images; see python dataset_tool.py --help for more information. Alternatively, the folder can also be used directly as a dataset, without running it through dataset_tool.py first, but doing so may lead to suboptimal performance.


FFHQ: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/stylegan3#:~:text=Flickr-Faces-HQ%20dataset) as 1024x1024 images and create a zip archive using dataset_tool.py: 
<!-- https://github.com/NVlabs/stylegan3#:~:text=Flickr-Faces-HQ%20dataset -->

https://github.com/NVlabs/ffhq-dataset


## Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-1024x1024.zip

## Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/images1024x1024 --dest=~/datasets/ffhq-256x256.zip \
    --resolution=256x256
See the FFHQ README for information on how to obtain the unaligned FFHQ dataset images. Use the same steps as above to create a ZIP archive for training and validation.

<!-- # How to run StyleGAN3 to generate synthetic images -->

# References
https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3

https://github.com/NVlabs/stylegan3
