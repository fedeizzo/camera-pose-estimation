# Camera pose estimation
This repo contains some implementations of relative and absolute camera pose estimation.

## Absolute pose estimation models
Currently absolute pose estimation models are:

- PoseNet
- MapNet

## Relative pose estimation models
Currently relative pose estimation models are:

- MeNet

## Sources
Sources of the techniques used are: (TODO)

- paper Colmap
- paper that explains all camera pose estimation attempts during the time
- paper PoseNet
- paper MapNet
- paper MeNet

## Dataset
Datasets are created using [Colmap](https://colmap.github.io/), this tool digests a video and compute the sparse and/or dense reconstruction of the environment. A intermediate step of this process is the camera pose estimation.

The positions computed by Colmap are well accurate but the process takes many times and resources.

## Goal
Given a set of videos sampled with a specific framerate is possible to generate labeled datasets that can be used to train a supervised model, once the training process for the model is completed it can be used as an effecient campera pose estimator.
