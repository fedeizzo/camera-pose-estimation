#!/usr/bin/env bash
project_root=$(git rev-parse --show-toplevel)
rclone sync ${project_root}/data drive-uni:SIV/data --progress
