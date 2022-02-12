# MoDeRNN: Towards Fine-grained Motion Details for Spatiotemporal Predictive Learning

## Abstract
Spatiotemporal predictive learning (ST-PL) aims at predicting the subsequent frames via limited observed sequences, and it has broad applications in the real world. However, learning representative spatiotemporal features for prediction is challenging. Moreover, chaotic uncertainty among consecutive frames exacerbates the difficulty in long-term prediction. This paper concentrates on improving prediction quality by enhancing the correspondence between the previous context and the current state. We carefully design Detail Context Block (DCB) to extract fine-grained details and improve the isolated correlation between upper context state and current input state. We integrate DCB with standard ConvLSTM and introduce Motion Details RNN (MoDeRNN) to capture fine-grained spatiotemporal features and improve the expression of latent states of RNNs to achieve significant quality. Experiments on Moving MNIST and Typhoon datasets demonstrate the effectiveness of the proposed method. MoDeRNN outperforms existing state-of-the-art techniques qualitatively and quantitatively with lower computation loads.



## Datasets
<!-- ## Pre-trained Models and Datasets -->
<!-- Pretrained Model will be released soon! -->
Moving MNIST dataset is avilliable at [here](https://drive.google.com/drive/folders/1Dl0WcevBRSsLn6KYJ7-zxMjLqo1S7WVr?usp=sharing):

## Setup
All code is developed and tested on a Nvidia RTX2080Ti the following environment.
- Ubuntu18.04.1
- CUDA 10.0
- cuDNN 7.6.4
- Python 3.8.5
- h5py 2.10.0
- imageio 2.9.0
- numpy 1.19.4
- opencv-python 4.4.0.46
- pandas 1.1.5
- pillow 8.0.1
- scikit-image 0.17.2
- scipy 1.5.4
- torch 1.7.1
- torchvision 0.8.2
- tqdm 4.51.0


## Train
### Train in Moving MNIST
Use the `train_mmnist.sh` script to train the model. To train the default model on Moving MNIST simply use:
```shell
sh train_mmnist.sh
```
You might want to change the `--data_root` which point to paths on your system of the data.

```
python train.py \
--model 'MoDeRNN' \
--dataset 'mmnist' \
--data_root './data/Moving_MNIST' \
--lr 0.001 \
--batch_size 32 \
--epoch_size 200 \
--input_nc 1 \
--output_nc 1 \
--load_size 720 \
--image_width 64 \
--image_height 64 \
--patch_size 4 \
--rnn_size 64 \
--rnn_nlayer 4 \
--filter_size 3 \
--seq_len 10 \
--pre_len 10 \
--eval_len 10 \
--criterion 'MSE&L1' \
--lr_policy 'cosine' \
--total_epoch 400 \
--data_threads 4 \
--optimizer adamw
```

## BibTeX
If you find this repository useful in your research, please consider citing the following paper:
```
@article{chai2021modernn,
  title={MoDeRNN: Towards Fine-grained Motion Details for Spatiotemporal Predictive Learning},
  author={Chai, Zenghao and Xu, Zhengzhuo and Yuan, Chun},
  journal={arXiv preprint arXiv:2110.12978},
  year={2021}
}
```


## Questions
If you have any questions or problems regarding the code or paper do not hesitate to contact us.