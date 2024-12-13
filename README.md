# Transformer_alphabet_ordering

Alphabet Sequence Reordering with a Transformer

This repository contains code for training and evaluating a Transformer-based model on a simple task: reordering sequences of alphabets into the correct alphabetical order. Although the problem is relatively straightforward, it serves as a hands-on example of building and using a Transformer in PyTorch.

Overview

The project demonstrates how to:

	•	Prepare a synthetic dataset of scrambled alphabet sequences.
 
	•	Build a vocabulary and encode tokens into integer representations.
 
	•	Define a Transformer model (encoder-decoder architecture) for sequence-to-sequence tasks.
 
	•	Train the model to learn the correct alphabetical ordering of input sequences.
 
	•	Evaluate the trained model on new inputs and observe its predictions.

 

File Structure


global_name_space.py:
Contains global configurations and hyperparameter definitions, including arguments for model size, number of layers, training epochs, batch size, and other options.


data_preparation.py:
Implements data generation and preprocessing functions:

	•	generate_alphabet_data(): Creates synthetic pairs of sequences and their correctly sorted counterparts.
 
	•	build_vocab_alphabet(): Builds a vocabulary mapping tokens (characters and special tokens like <sos>, <eos>, <pad>) to integer indices.
 
	•	preprocess_alphabet_data(): Prepares the dataset for training by converting tokens to their corresponding indices and handling padding.

 
main.py:
Defines the Transformer model’s components:

	•	SelfAttention: Implements multi-head self-attention.
 
	•	TransformerBlock: A single encoder/decoder block with self-attention and feed-forward layers.
 
	•	Encoder and Decoder: Stacks multiple TransformerBlocks and handles positional embeddings.
 
	•	Transformer: Combines the encoder and decoder to form the complete seq2seq model.
 
Also includes code to load a saved model checkpoint and perform inference on test inputs.


train.py:
Handles the training loop:

	•	Loads the vocabulary and the dataset.
 
	•	Initializes the Transformer model.
 
	•	Defines the optimizer and loss function.
 
	•	Runs the training epochs, periodically saves checkpoints, and reports training loss.
