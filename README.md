# LLMOps

This repository provides an hour-long seminar about distributed training large language models

## Requirements
```
sudo apt install libopenmpi-dev  # for mpi4py
pip install -r requirements.txt
```

## How to Train
```
python3 1_make_model.py
python3 2_make_dataset.py
. run.sh
```

## How to Inference
```
python3 4_inference.py
```
