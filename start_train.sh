#!/bin/bash
echo "hello world"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate uav_ai
conda info
python -V
# python main.py
read -p "Press enter to continue"
gnome-terminal &   # launch a new terminal
kill $PPID         # kill this

