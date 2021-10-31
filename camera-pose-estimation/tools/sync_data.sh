#!/usr/bin/env bash
project_root=$(git rev-parse --show-toplevel)
rclone copy ${project_root}/data drive-uni:SIV/data --progress
rclone copy ${project_root}/papers drive-uni:SIV/papers --progress
rclone copy drive-uni:SIV/data ${project_root}/data --progress
rclone copy drive-uni:SIV/papers ${project_root}/papers --progress
