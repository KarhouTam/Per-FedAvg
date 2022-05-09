# Per-FedAvg
PyTorch Implementation of [Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html) [NIPS 2020]

For simulating Non-I.I.D scenario, the dataset is split by labels and each client has only **two** classes of data.

It’s just a toy demo for demonstrating algorithm, so take it easy. 🤣

## Requirements
torch~=1.10.2

path~=16.4.0

numpy~=1.21.2

fedlab~=1.1.4

torchvision~=0.11.3

rich~=12.2.0


## Preprocess dataset

MNIST and CIFAR-10 is prepared.🌟

```
cd data/$dataset; python preprocess.py
```

The way of preprocessing is adjustable, more details in each dataset folder's `preprocess.py`.



## Run the experiment

Before run the experiment, please make sure that the dataset is downloaded and preprocessed already.

It’s so simple.🤪 

```
python main.py
```



## Hyperparameters

`--global_epochs`: Num of communication rounds. Default: `200`

`--local_epochs`: Num of local training rounds. Default: `4`

`--pers_epochs`: Num of personalization rounds (while in evaluation phase). Default: `200`

`--dataset`: Name of experiment dataset. Default: `mnist`

`--fraction`: Fraction of clients for training in all clients. Default: `0.9`

`--client_num_per_round`: Num of clients that participating training at each communication round. Default: `5`

`--alpha`: Learning rate $\alpha$ in paper. Default: `0.01`

`--beta`: Learning rate $\beta$ in paper. Default: `0.001`

`--gpu`: Non-zero value for using CUDA; `0` for using CPU. Default: `1`

`--batch_size`: Batch size of client local dataset. Default: 40.

`--eval_while_training`: Non-zero value for performing evaluation while in training phase. Default: `1`

`--valset_ratio`: Fraction of validation set in client dataset. Default: `0.1`

`--hf`: Non-zero value for performing Per-FedAvg(HF); `0` for Per-FedAvg(FO). Default: `1`

`--seed`: Random seed for init model parameters and selected clients.
