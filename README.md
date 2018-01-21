# CrabNet
Link prediction in Knowledge Bases using text classification.

## Pre-request libraries
- python3
- tensorflow

### Requirements
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quick-start) is used for reliable GPU support in the containers. This is an extension to Docker and can be easily installed with just two commands.
To run the networks, you need an Nvidia GPU with >1GB of memory (at least Kepler).

## 1. Data preparation (FB15K dataset example)
Datasets are required in the following format, containing five files:
triple2id.txt: triples ids file, the first line is the number of triples for training. Then the follow lines are all in the format (e1, e2, rel).

entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

ent_embeddings_transE.txt: all entity features (128 dimension), one per line.

rel_embeddings_transE.txt: all relation features (128 dimension), one per line.

You can find all the FB15K data in the data folder.

## 2. Train and test the model
Run the model.py file with suitable papermeters, which you can find inside the file.

## 3. Results:
The result is evaluated based on the FB15K dataset.


