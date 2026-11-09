#!/bin/sh
# Download the LUAR TorchScript checkpoint used for evaluation (utility/LUAR.pth).
# This is the checkpoint distributed with the STYLL/TinyStyler authorship
# evaluation scripts. Requires `pip install gdown`.
set -ex
gdown "https://drive.google.com/file/d/13lLj2fK8IycK_EYPJeO7TEmx9knFe0Cv/view?usp=sharing" --fuzzy -O "$(dirname "$0")/LUAR.pth"
