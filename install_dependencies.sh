#!/bin/bash

source /home/groups/rbaltman/tartici/miniconda3/etc/profile.d/conda.sh
# Load appropriate compiler module if needed
module load gcc/10.3.0
export LD_LIBRARY_PATH=/path/to/gcc/10.3.0/lib64:$LD_LIBRARY_PATH
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Create and activate env
conda env create -f environment.yml
source /home/groups/rbaltman/tartici/miniconda3/etc/profile.d/conda.sh
conda activate parse
