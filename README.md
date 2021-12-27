# Camera pose estimation
This repo contains some implementations of relative and absolute camera pose estimation.

![Introduction example](./docs/imgs/introduction_example.png)

Full documentation can be found in file [main.pdf](./docs/main.pdf)

## Absolute pose estimation models
Currently absolute pose estimation models are:

- PoseNet
- MapNet

### PoseNet
In order to run PoseNet it is required to create a config files:
- [train](./camera-pose-estimation/model/posenet_train.ini.sample)
- [test](./camera-pose-estimation/model/posenet_test.ini.sample)

*Train*
```bash
python ./camera-pose-estimation/model/run.py --config ./camera-pose-estimation/model/posenet_train.ini --train
```

*Test*
```bash
python ./camera-pose-estimation/model/run.py --config ./camera-pose-estimation/model/posenet_test.ini --test
```

#### Config docs
*Train*
```toml
[environment]
# name of the experiment for net weights dir and aim
experiment_name = "posenet"
# name of the run for net weights dir and aim
run_name = "resnet152_weighted_custom_2"
seed = 0
epochs = 50

[model]
name = "posenet"
# value that can be adjust based on the representation used
# for the orientation. In this case 3 for pose and 4 for quaternion
outputs = 7

[dataloader]
# save preprocessed dataset according to data folder structure.
# This reduces the dataset loading time on each run
save_processed_dataset = true
num_workers=4
batch_size = 128

[loss]
# available losses: mse, L1Loss, SmoothL1Loss, weighted
type = "L1Loss"
# weights used only with weighted loss
weights = [1, 1, 1, 1, 1, 1, 1]

[optimizer]
name = "SGD"
lr = 0.1
momentum = 0.9

[scheduler]
name = "StepLR"
step_size = 20
gamma = 0.1

[paths]
# datasets can be a csv file (in this case it is computed a slow preprocessing)
# or a folder in which a preprocessed dataset is saved (faster loading)
# example:
#    train_dataset = "/path/to/train.csv"
# or
#    train_dataset = "/path/to/data/my_dataset/processed_dataset/train"
train_dataset = "path"
validation_dataset = "path"
test_dataset = "path"
# images, net_weights_dir and aim_dir are directories
images = "path"
net_weights_dir = "path"
aim_dir = "path"
```

*Test*
```toml
[environment]
# experiment and run names used during train phase
# in this way the script will load the train config file
experiment_name = "posenet"
run_name = "resnet152"

[dataloader]
# save preprocessed dataset according to data folder structure.
# This reduces the dataset loading time on each run
save_processed_dataset = true

[paths]
# datasets can be a csv file (in this case it is computed a slow preprocessing)
# or a folder in which a preprocessed dataset is saved (faster loading)
# example:
#    test_dataset = "/path/to/test.csv"
# or
#    test_dataset = "/path/to/data/my_dataset/processed_dataset/test"
test_dataset = "path"
# targets and predictions are csv files where targets and predictions is saved
targets = "path"
predictions = "path"
# images and net_weights_dir are directories
images = "path"
net_weights_dir = "path"
```

### PoseNet
In order to run PoseNet it is required to create a config files:
- [train](./camera-pose-estimation/model/mapnet_train.ini.sample)
- [test](./camera-pose-estimation/model/mapnet_test.ini.sample)

*Train*
```bash
python ./camera-pose-estimation/model/run.py --config ./camera-pose-estimation/model/mapnet_train.ini --train
```

*Test*
```bash
python ./camera-pose-estimation/model/run.py --config ./camera-pose-estimation/model/mapnet_test.ini --test
```

#### Config docs
*Train*
```toml
[environment]
# name of the experiment for net weights dir and aim
experiment_name = "posenet"
# name of the run for net weights dir and aim
run_name = "resnet152_weighted_custom_2"
seed = 0
epochs = 50

[model]
name = "mapnet"
feature_dimension = 2048
dropout_rate = 0.5

[dataloader]
num_workers = 4
batch_size = 64
# value used for torchvision jitter transform
color_jitter = 0.5
# defines how many images are ingested by the relative pose criterion
step = 5
# defines the distance between two images ingested by the relative pose criterion
skip = 5
# save preprocessed dataset according to data folder structure.
# This reduces the dataset loading time on each run
save_processed_dataset = true

# required only with 7scenes dataset
# sequences = {"train": "seq-02", "validation": "seq-03"}

[loss]
# parameters of the custom mapnet criterion, please refer to the
# full documentation pdf file in docs folder
type = "mapnet_criterion"
beta = 0
gamma = -3.0
learn_beta = true
learn_gamma = true

[optimizer]
name = "SGD"
lr = 0.05
momentum = 0.9

[scheduler]
name = "StepLR"
step_size = 60
gamma = 0.5

[paths]
# datasets can be a csv file (in this case it is computed a slow preprocessing)
# or a folder in which a preprocessed dataset is saved (faster loading)
# example:
#    test_dataset = "/path/to/test.csv"
# or
#    test_dataset = "/path/to/data/my_dataset/processed_dataset/test"
train_dataset = "path"
validation_dataset = "path"
test_dataset = "path"
# images, net_weights_dir and aim_dir are directories
net_weights_dir = "path"
aim_dir = "path"
images = "path"
```

*Test*
```toml
[environment]
# experiment and run names used during train phase
# in this way the script will load the train config file
experiment_name = "posenet"
run_name = "resnet152"

[dataloader]
# save preprocessed dataset according to data folder structure.
# This reduces the dataset loading time on each run
save_processed_dataset = true

[paths]
# datasets can be a csv file (in this case it is computed a slow preprocessing)
# or a folder in which a preprocessed dataset is saved (faster loading)
# example:
#    test_dataset = "/path/to/test.csv"
# or
#    test_dataset = "/path/to/data/my_dataset/processed_dataset/test"
test_dataset = "path"
# targets and predictions are csv files where targets and predictions is saved
targets = "path"
predictions = "path"
# images and net_weights_dir are directories
images = "path"
net_weights_dir = "path"
```

## Relative pose estimation models
Currently relative pose estimation models are:

- MeNet

## Dataset
Datasets are created using [Colmap](https://colmap.github.io/), this tool digests a video and compute the sparse and/or dense reconstruction of the environment. A intermediate step of this process is the camera pose estimation.

The positions computed by Colmap are well accurate but the process takes many times and resources.

### Video to dataset
The script [video to dataset](./camera-pose-estimation/tools/video_to_dataset.sh) takes a video and split it into multiple images that are used to compute a sparse or dense model using Colmap. Once the reconstruction is done the script extract poses of the cameras.
```bash
usage: video_to_dataset.sh [-h] -v VIDEO -o OUTPUT_PATH [-f FRAMES] [-c CAMERA_REFENCE_PATH]
                           [-n NUM_THREADS] [-q {low,medium,high,extreme}] [-t {sparse,dense}]

Convert video to a dataset

optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to the video
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output path where images are saved
  -f FRAMES, --frames FRAMES
                        Number of frames to extract
  -c CAMERA_REFENCE_PATH, --camera_refence_path CAMERA_REFENCE_PATH
                        Path where camera db values is saved
  -n NUM_THREADS, --num_threads NUM_THREADS
                        Number of threads to use (default all threads)
  -q {low,medium,high,extreme}, --quality {low,medium,high,extreme}
                        Quality of colmap reconstruction
  -t {sparse,dense}, --type {sparse,dense}
```

### Cross validation split
Scripts [split abasolute dataset](./camera-pose-estimation/tools/split_absolute_dataset.sh) and [split relative dataset](./camera-pose-estimation/tools/split_relative_dataset.sh) split dataset into train, test and validation files given a folder created with the video to dataset script.

```bash
usage: split_absolute_dataset.sh [-h] -i INPUT_PATH [-n] [-t TRAIN_SPLIT] [-v VALIDATION_SPLIT]
                                 [-e TEST_SPLIT]

Convert video to a dataset

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Input path where images and models are saved
  -n, --no_normalization
                        Block normalization process
  -t TRAIN_SPLIT, --train_split TRAIN_SPLIT
                        Number of samples for train phase
  -v VALIDATION_SPLIT, --validation_split VALIDATION_SPLIT
                        Number of samples for validation phase
  -e TEST_SPLIT, --test_split TEST_SPLIT
                        Number of samples for test phase
```

### Data folder structure
Given a video the folder specificied by `OUTPUT_PATH` will presents the following structure:
```bash
 .
├──  imgs
│  ├──  0000.png
│  ├──  ........
│  └──  XXXX.png
├──  processed_dataset
│  ├──  test
│  │  ├──  dataset_X.pt
│  │  └──  dataset_Y.pt
│  ├──  train
│  │  ├──  dataset_X.pt
│  │  └──  dataset_Y.pt
│  └──  validation
│     ├──  dataset_X.pt
│     └──  dataset_Y.pt
├──  workspace
│  ├──  sparse
│  │  └──  0
│  │     ├──  cameras.bin
│  │     ├──  images.bin
│  │     ├──  points3D.bin
│  │     └──  project.ini
│  └──  database.db
├──  points3D.csv
├──  positions.csv
├──  test.csv
├──  train.csv
└──  validation.csv
```

- imgs folder contains all frames
- processed\_dataset folder contains the dataset processed with the absolute pose dataset (see model section for more information)
- workspace contains models created by Colmap
- points3D.csv contains all features extracted by Colmap
- positions.csv contains all camera poses extracted by Colmap
- train.csv, validation.csv and test.csv are files created with the split scripts

## Goal
Given a set of videos sampled with a specific framerate is possible to generate labeled datasets that can be used to train a supervised model, once the training process for the model is completed it can be used as an effecient campera pose estimator.
