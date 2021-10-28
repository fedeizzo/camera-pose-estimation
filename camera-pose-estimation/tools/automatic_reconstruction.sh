#!/usr/bin/env bash
[ -z $1 ] && echo 'usage: automatic_reconstructor.sh <output_folder> <input_folder> <quality>' && exit 1
[ -z $2 ] && echo 'usage: automatic_reconstructor.sh <output_folder> <input_folder> <quality>' && exit 1
[ -z $3 ] && echo 'usage: automatic_reconstructor.sh <output_folder> <input_folder> <quality>' && exit 1
colmap automatic_reconstructor \
    --workspace_path $1 \
    --image_path $2 \
    --dense yes \
    --sparse no \
    --quality $3 \
    --use_gpu 0 \
    --num_threads 16 \
    --data_type video \
