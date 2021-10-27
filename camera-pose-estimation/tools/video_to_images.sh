#!/usr/bin/env bash
[ -z $1 ] && echo 'usage: video_to_images.sh <video> <output_folder> <fps>' && exit 1
[ -z $2 ] && echo 'usage: video_to_images.sh <video> <output_folder> <fps>' && exit 1
[ -z $3 ] && echo 'usage: video_to_images.sh <video> <output_folder> <fps>' && exit 1
ffmpeg -i $1 -vf fps=$3 $2/out%d.png
