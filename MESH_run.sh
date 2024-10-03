#!/bin/bash
#python main.py anonymize_directory --model_file lib/models/03_iris_no_tesselation.ckpt --input_directory  ../datasets/MetaGaze_splitted/test_files --output_directory  ../datasets/Mesh_results/03_MG
python main.py anonymize_directory --model_file lib/models/03_iris_no_tesselation.ckpt --input_directory  ../datasets/DMD/frames --output_directory  ../datasets/Mesh_results/04_DMD
