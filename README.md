# Exploring the optimizers landscape - Comparing AdaBelief and Madgrad

This repository presents the code for the project of **Optimization for Machine Learning** course (EPFL, 2022), by **Kévin Faustini**, **Thomas Traversié** and **Luca Zampierin**.

In this project, we compare four optimizers: **Stochastic Gradient Descent with Momentum**, **Adam** [1], **Madgrad** [2] and **AdaBelief** [3]. This comparison is done on two optimization problems: **Image Classification** on CIFAR10 and **Image Denoising**. We proceed with both default hyper-parameters and tuned hyper-parameters. Hyper-parameter tuning is achieved thanks to Tree-Parzen Estimator algorithm implemented in the Optuna framework [4,5]. The experiments are run through five different seeds to dampen the effect of randomness.

Parts of the code for Image Classification are adapted from pytorch examples (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).


## Organization of the repository

- The **Image Classification** problem is done in the notebook `OptML_image_classification.ipynb`. The explanations of the code are self-encapsulated in the notebook. Hence the notebook contains steps for the set-up of the environement and the loading af the data. Then we proceed to the training of the ResNet18 model [6] with the four optimizers using default parameters. We tune the hyperparameter and achieve the training with tuned hyperparameters. Finally, we plot the results.

- The **Denoising Problem** is done in the notebook `OptML_denoising.ipynb`. The explanations of the code are self-encapsulated in the notebook. Hence the notebook contains steps for the set-up of the environement and the loading af the data. Then we proceed to the training of the UNet model [7,8] with the four optimizers using default parameters. We tune the hyperparameter and achieve the training with tuned hyperparameters. Finally, we plot the results.

- The **results data** are available in the folder `results_data`. The format of the subfolders is the following: *[optimizer name]_[problem_name]*. *mix* stands for the combination of Madgrad and AdaBelief, *best* for the case with tuned hyperparameters, *cifar* is for Image Classification, *denoising* for Image Denoising, *acc* for the accuracy data and *loss* for the data of the loss. *episode_numer* stands for the number of the seed considered. In the subfolder `mix_cifar`, *T* represents the switching time between Madgrad and AdaBelief.

- The **plots** are available in the folder `plot`. In the name of the images, *default* is for the plot obtained with the defaut hyperparameters, and *tuned* is for the plot obtained with tuned hyperparameters.

- `hyperparameter_tuning_cifar` and `hyperparameter_tuning_denoising` are the saved trials of the tuning phase of `OptML_image_classification.ipynb` and `OptML_denoising.ipynb`.

## Downloading of the data for Image Denoising

The data used for Image Denoising was made available for the Deep Learning course (EE-559). It can be downloaded at https://drive.google.com/drive/u/2/folders/1CYsJ5gJkZWZAXJ1oQgUpGX7q5PxYEuNs (but only available with an EPFL account).


## Running the code

The code can be run using any `Jupyter Notebook` for instance. However, due to the long runtime, it is recommended to use `Google Colab` to have a GPU. 

1. Go to https://colab.research.google.com/
2. Import the Notebook
3. Configure the GPU: 
  - Click on `Edit`
  - Click on `Notebook settings`
  - Select `GPU` in the `Hardware accelerator` box and `Save` the new settings.


## References

[1] D. P. Kingma and J. L. Ba, "Adam: A method for stochastic optimization", *International Conference for Learning Representations*, 2015.

[2] A. Defazio and S. Jelassi, "Adaptivity without compromise: A momentumized, adaptive, dual averaged gradient method for stochastic optimization", 2021.

[3] J. Zhuang, T. Tang, Y. Ding, S. Tatikonda, N. Dvornek, X. Papademetris, and J. Duncan, "Adabelief optimizer: Adapting stepsizes by the belief in observed gradients", *Conference on Neural Information Processing Systems*, 2020.

[4] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kegl, "Algorithms for hyper-parameter optimization", *Advances in neural information processing systems*, vol. 24, 2011.

[5] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A nextgeneration hyperparameter optimization framework", in *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*, 2019, pp. 2623–2631.

[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition", *CoRR*, vol. abs/1512.03385, 2015. [Online]. Available: http://arxiv.org/abs/1512.03385

[7] J. Lehtinen, J. Munkberg, J. Hasselgren, S. Laine, T. Karras, M. Aittala, and T. Aila, "Noise2noise: Learning image restoration without clean data", arXiv preprint arXiv:1803.04189, 2018.

[8] O. Ronneberger, P. Fischer, and T. Brox, "U-net: Convolutional networks for biomedical image segmentation", in *International Conference on Medical image computing and computer-assisted intervention*. Springer, 2015, pp. 234–241.
