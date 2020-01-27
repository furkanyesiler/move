# MOVE: Musically-motivated Version Embeddings
This repository contains the training and evaluation code of our state-of-the-art version identification system along with pre-trained models. For a detailed explanation of our design decisions and the general pipeline, please refer to our [publication](https://arxiv.org/abs/1910.12551).

> Furkan Yesiler, Joan Serrà and Emilia Gómez, "Accurate and scalable version identification using musically-motivated embeddings," in Proc. of the IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), 2020.

<p align="center"> 
    <img src="https://user-images.githubusercontent.com/32430027/73191022-23e59580-4127-11ea-90aa-125f8ff03028.png" alt="MOVE"/>
</p>

## Using the code
Below, we specify some use cases and explain the important steps you need to follow for using the code in this repository.

* Evaluating the pre-trained models on Da-TACOS
* Evaluating the pre-trained models on a private dataset
* Training a model with a private dataset

### 1 - Evaluating the pre-trained models on Da-TACOS benchmark subset
To facilitate the benchmarking process and to present a pipeline for evaluating the pre-trained MOVE models, we have prepared [benchmark_da-tacos.py](https://github.com/furkanyesiler/move/blob/master/benchmark_da-tacos.py). To use the script, you can follow the steps below:

#### 1.1 - Requirements
* Python 3.6+
* Create a virtual enviroment and install requirements
```bash
git clone https://github.com/furkanyesiler/move.git
cd move
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_benchmark.txt
```

#### 1.2 - Running benchmark_da-tacos.py
After creating the virtual environment and installing the required packages, you can simply run
```bash
python benchmark_da-tacos.py --unpack --remove
```
```
usage: benchmark_da-tacos.py [-h] [--outputdir OUTPUTDIR] [--unpack]
                             [--remove]

Downloading and preprocessing cremaPCP features of the Da-TACOS benchmark
subset

optional arguments:
  -h, --help            show this help message and exit
  --outputdir OUTPUTDIR
                        Directory to store the dataset (default: ./data)
  --unpack              Unpack the zip files (default: False)
  --remove              Remove zip files after unpacking (default: False)
```

This script downloads the metadata and the cremaPCP features of the Da-TACOS benchmark subset, and preprocesses them to work with our evaluation setting. Specifically, after downloading the files:
* it downsamples the cremaPCP features by 8, 
* reshapes them from Tx12 to 1x23xT (for the intuition behind this step, you can check our paper), 
* stores them in a dictionary which is saved as a `.pt` file,
* creates ground truth annotations to be used by our evaluation function.

Both the data and the ground truth annotations (named `benchmark_crema.pt` and `ytrue_benchmark.pt`, respectively) are stored in the `data` folder.

### 2 - Evaluating the pre-trained models on a private dataset
For this use case, we would like to point out a number of requirements you must follow.

#### 2.1 - Requirements
* Python 3.6+
* Create a virtual enviroment and install requirements
```bash
git clone https://github.com/furkanyesiler/move.git
cd move
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2.2 - Setting up the dataset and annotations
MOVE was trained using cremaPCP features, and therefore, it may underperform with other Pitch Class Profile (PCP, or chroma) variants. To extract cremaPCP features for your audio collection, please refer to [acoss](https://github.com/furkanyesiler/acoss).

cremaPCP features used for training are created using non-overlapping frames and a hop size of 4096 on audio tracks sampled at 44.1 kHz (about 93 ms per frame).

After obtaining cremaPCP features for your dataset, you should cast them as a torch.Tensor, and reshape them from Tx12 to 1x23xT. For this step, you can use the following code snippet:

```python
import numpy as np
import torch

# this variable represents your cremaPCP feature
cremaPCP = np.random.rand(100, 12)

# this is casting the crema feature to a torch.Tensor type
cremaPCP_tensor = torch.from_numpy(cremaPCP).t()

# this is the resulting cremaPCP feature
cremaPCP_reshaped = torch.cat((cremaPCP_tensor, cremaPCP_tensor))[:23].unsqueeze(0)

```

#### 1.3 - Running the evaluation script
After the features are downloaded and preprocessed, you can use the script below to evaluate the pre-trained MOVE model on the Da-TACOS benchmark subset:

```bash
python move_main.py -rt test --dataset 1
```
or
```bash
python move_main.py -rt test --dataset 1 -emb 4000
```

##### 2.2.1 - Dataset file

When the cremaPCP features per song are ready, you need to create a dataset file and a ground truth annotations file. The dataset file should be a python dictionary with 2 keys: `data` and `labels`. Each key (i.e. 'data' or 'labels') should point to a python list which contains the respective cremaPCP features and label of that track. Specifically, let `dataset_dict` be our dataset dictionary, and `dataset_dict['data']` and `dataset_dict['labels']` be our lists. The label of the song `dataset_dict['data'][42]` should be `dataset_dict['labels'][42]`. Finally, the dataset file should be saved under `data` folder, and should be named `benchmark_crema.pt`. An example code is shown below:

```python
import os

root_dir = '/your/root/directory/of/move'

data = []
labels = []
for i in range(dataset_size):
	cremaPCP = load_cremaPCP(i)  # loading the cremaPCP features for the ith song of your dataset
	label = load_label(i)  # loading the label of the ith song of your dataset

	data.append(cremaPCP)
	labels.append(label)

dataset_dict = {'data': data, 'labels': label}

torch.save(dataset_dict, os.path.join(root_dir, 'data', 'benchmark_crema.pt'))

```

##### 2.2.2 - Annotations file

When your dataset file ('benchmark_crema.pt') is ready, you have to create a ground truth annotations file which is stored in `data` folder, and should be named `ytrue_benchmark.pt`. This file should be a torch.Tensor with the shape NxN (N is the size of your dataset). Finally, the diagonal of this matrix should be 0. You can find an example code below:

```python
import os

import torch

data_dir = '/your/root/directory/of/move/data/'

labels = torch.load(os.path.join(data_dir, 'benchmark_crema.pt'))['labels']

ytrue = []

for i in range(len(labels)):
	main_label = labels[i]  # label of the ith song
	sub_ytrue = []
	for j in range(len(labels)):
		if labels[j] == main_label and i!= j:  # checking whether the ith and jth song has the same label
			sub_ytrue.append(1)
		else:
			sub_ytrue.append(0)
	ytrue.append(sub_ytrue)

ytrue = torch.Tensor(ytrue)
torch.save(ytrue, os.path.join(data_dir, 'ytrue_benchmark.pt'))
```

#### 2.3 - Running the evaluation script
After you prepared your dataset and annotation files, you can use the script below to evaluate the pre-trained MOVE model on your dataset:

```bash
python move_main.py -rt test --dataset 1
```
or
```bash
python move_main.py -rt test --dataset 1 -emb 4000
```


### 3 - Training a model with a private dataset
For training MOVE with a private dataset, you should follow the steps below:

#### 3.1 - Requirements
* Python 3.6+
* Create a virtual enviroment and install requirements
```bash
git clone https://github.com/furkanyesiler/move.git
cd move
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3.2 - Setting up the data and annotations
The main requirement for the training is to create dataset file(s) and an annotation file for the validation set. For this, you should extract cremaPCP features for the songs in your dataset, and apply the preprocessing steps described 
at 2.2.

The dataset files should be created with the same structure as explained in 2.2.1. They should be python dictionaries with two keys (i.e. `'data'` and `'labels'`), and should be placed under `data` folder. 

If your training data includes more than one file, then you should use the following naming convention:
```
If your dataset is named 'my_training_set', and is divided into 3 files, you should name them:
my_training_set_1.pt
my_training_set_2.pt
my_training_set_3.pt

In this case, when you run move_main.py, you should use the following arguments:

python move_main.py --train_path my_training_set --chunks 3

```

Current version of our code does not support using more than one file for the validation set.

The training procedure of MOVE tracks Mean Average Precision (MAP) score on the validation set for each epoch. To compute MAP, you need to provide an annotations file under the `data` folder. You can see an example of how to create such file at 2.2.2. The name of your annotations file for your validation set should be `ytrue_validation.pt`.

#### 3.3 - Running the training code
After preparing the dataset and annotations files, you can run the training procedure with the following command:

```bash
python move_main.py
```
```
usage: move_main.py [-h] [-rt {train,test}] [-tp TRAIN_PATH] [-ch CHUNKS]
                    [-vp VAL_PATH] [-sm {0,1}] [-ss {0,1}] [-rs RANDOM_SEED]
                    [-noe NUM_OF_EPOCHS] [-m {0,1}] [-emb EMB_SIZE]
                    [-sum {0,1,2,3,4}] [-fa {0,1,2,3}] [-lr LEARNING_RATE]
                    [-lrs {0,1,2}] [-lrsf LRSCH_FACTOR] [-mo MOMENTUM]
                    [-pl PATCH_LEN] [-nol NUM_OF_LABELS] [-da {0,1}]
                    [-nd {0,1}] [-ms {0,1,2}] [-ma MARGIN] [-ytc {0,1}]
                    [-d {0,1,2}] [-dn DATASET_NAME]

Training code of MOVE

optional arguments:
  -h, --help            show this help message and exit
  -rt {train,test}, --run_type {train,test}
                        Whether to run train or test script
  -tp TRAIN_PATH, --train_path TRAIN_PATH
                        Path for training data. If more than one file are
                        used, write only the common part
  -ch CHUNKS, --chunks CHUNKS
                        Number of chunks for training set
  -vp VAL_PATH, --val_path VAL_PATH
                        Path for validation data
  -sm {0,1}, --save_model {0,1}
                        1 for saving the trained model, 0 for otherwise
  -ss {0,1}, --save_summary {0,1}
                        1 for saving the training log, 0 for otherwise
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        Random seed
  -noe NUM_OF_EPOCHS, --num_of_epochs NUM_OF_EPOCHS
                        Number of epochs for training
  -m {0,1}, --model_type {0,1}
                        0 for MOVE, 1 for MOVE without pitch transposition
  -emb EMB_SIZE, --emb_size EMB_SIZE
                        Size of the final embeddings
  -sum {0,1,2,3,4}, --sum_method {0,1,2,3,4}
                        0 for max-pool, 1 for mean-pool, 2 for autopool, 3 for
                        multi-channel attention, 4 for multi-channel adaptive
                        attention
  -fa {0,1,2,3}, --final_activation {0,1,2,3}
                        0 for no activation, 1 for sigmoid, 2 for tanh, 3 for
                        batch norm
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Initial learning rate
  -lrs {0,1,2}, --lr_schedule {0,1,2}
                        0 for no lr_schedule, 1 for decreasing lr at epoch 80,
                        2 for decreasing lr at epochs [80, 100]
  -lrsf LRSCH_FACTOR, --lrsch_factor LRSCH_FACTOR
                        Factor for lr scheduler
  -mo MOMENTUM, --momentum MOMENTUM
                        Value for momentum parameter for SGD
  -pl PATCH_LEN, --patch_len PATCH_LEN
                        Size of the input len in time dimension
  -nol NUM_OF_LABELS, --num_of_labels NUM_OF_LABELS
                        Number of cliques per batch for triplet mining
  -da {0,1}, --data_aug {0,1}
                        0 for no data aug, 1 using it
  -nd {0,1}, --norm_dist {0,1}
                        1 for normalizing the distance, 0 for avoiding it
  -ms {0,1,2}, --mining_strategy {0,1,2}
                        0 for only random, 1 for only semi-hard, 2 for only
                        hard
  -ma MARGIN, --margin MARGIN
                        Margin for triplet loss
  -ytc {0,1}, --ytc_labels {0,1}
                        0 for using full training data, 1 for removing
                        overlapping labels with ytc
  -d {0,1,2}, --dataset {0,1,2}
                        Choosing evaluation set for testing. 0 for move
                        validation, 1 for test on da-tacos, 2 for test on ytc
  -dn DATASET_NAME, --dataset_name DATASET_NAME
                        Specifying a dataset name for evaluation. The dataset
                        must be located in the data folder

```

## Questions
For any questions you may have, feel free to create an issue or contact [me](mailto:furkan.yesiler@upf.edu).

## License
The code in this repository is licensed under [Affero GPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html).

## References
Please cite our reference if you plan to use the code in this repository:
```
@inproceedings{yesiler2020,
    author = "Furkan Yesiler and Joan Serrà and Emilia Gómez",
    title = "Accurate and scalable version identification using musically-motivated embeddings",
    booktitle = "Proc. of the IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP)",
    year = "2020"
}
```

## Acknowledgments

This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068 (MIP-Frontiers).

This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 770376 (TROMPA).

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" height="64" hspace="20">
