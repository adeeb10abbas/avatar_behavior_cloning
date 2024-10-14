#!/bin/bash
set -eux

cd $(dirname $0)/../data
mkdir -p pkl
cd pkl

if [[ -! f 2024-07-26-21-15-44.pkl ]]; then
  gdown --fuzzy https://drive.google.com/file/d/1dafKcCxek7zJbCX3qVCPgglh8QT7xY9J/view?usp=drive_link
fi
