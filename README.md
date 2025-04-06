## Sentence Gestalt Models (SGM)

A PyTorch Lightning implementation of a simplified Structured Gestalt Model (SGM) for structured role-filler prediction in language data. This model learns to encode input sentences into context-rich representations and predict complex structured outputs involving semantic roles and event frames.


## Key Features

- SGMnet: Combines encoder and decoder functionality in a unified LSTM architecture.
- Role-Filler Training: Learns mappings between input words and semantic roles using probe-based supervision.
- Flexible Inputs: Trains on variable-length sentences with customizable number of probes and frames.
- Lightning Integration: Scalable and distributed training with PyTorch Lightning.
- Embeddings: Compatible with pre-trained FastText vectors.


## Project Structure

- `SGM.py`: Core model (SGMnet) using PyTorch Lightning.
- `dataset.py`: Data loading and preprocessing for SGM-style sentence/probe/target triplets.
- `trainer_sgm.py`: Script for training, continuing training, or fine-tuning.
- `bash_sgm.sh`: Shell script to run training with predefined arguments.
- `__init__.py`: Module setup.


## Usage

# Train from Scratch

python trainer_sgm.py \
  --id '01' \
  --min_words 5 --max_words 15 \
  --nr_frames 2 --nr_probes 8 \
  -i 600 -g 1200 -l 1 \
  --nr_workers 1 \
  --max_nr_files 3000 -e 1 \
  --lr 0.001 --dp 0.1 \
  --optimizer 'Adamax' \
  --data '../data_path/' \
  --log_directory '../TRAINED_MODELS/SGM/' \
  --continue_training 'False'

Or simply run:

bash bash_sgm.sh

You can resume or fine-tune training via: 

--continue_training or --fine_tune

# Requirements

pip install torch pytorch-lightning gensim numpy scipy

Ensure you also have:
- Pre-trained FastText `.txt` file (use `--embeddings`) -- in data/features/
- Properly formatted `.inlr.txt`, `.prblr.txt`, and `.outlr.txt` data files


## Model Architecture

SGMnet uses:
- Word embeddings → LSTM encoder (Gestalt state)
- Probes → MLP → Combined with LSTM state
- Outputs → Sigmoid over role-filler prediction vector

Each sentence is processed word-by-word, with predictions structured around semantic frame-role probing.


## Notes

- Created by A. Lopopolo, 2021.
- Designed to test SGM-based architectures in a flexible, scalable way.


## Publications

This codebase was developed and used as part of the following research studies:

- A. Lopopolo, M. Rabovsky (2024). Tracking lexical and semantic prediction error underlying the N400 using artificial neural network models of sentence processing. Neurobiology of Language, 5(1), 136–166.

- M. Rabovsky, A. Lopopolo, D. J. Schad (2024). Interindividual differences in predicting words versus sentence meaning: Explaining N400 amplitudes using large-scale neural network models. Proceedings of the Annual Meeting of the Cognitive Science Society, 46, 2024

