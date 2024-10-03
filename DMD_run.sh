#!/bin/bash
python main.py anonymize_directory --model_file lib/models/msc/epoch=24-step=278750.ckpt --input_directory  ../datasets/DMD/frames --output_directory  ../datasets/DMD/results/Cel

python main.py anonymize_directory --model_file lib/models/msc/epoch=24-step=66150.ckpt --input_directory  ../datasets/DMD/frames --output_directory  ../datasets/DMD/results/GM


python main.py anonymize_directory --model_file lib/models/msc/epoch=49-step=557500.ckpt --input_directory  ../datasets/DMD/frames --output_directory  ../datasets/DMD/results/Cel+GM

python main.py anonymize_directory --model_file lib/models/msc/epoch=49-step=133848.ckpt --input_directory  ../datasets/DMD/frames --output_directory  ../datasets/DMD/results/GM+Cel
