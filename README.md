# Text-to-Speech Model Training

## **Overview**
This repository contains the code and resources for training a text-to-speech (TTS) model. The goal is to fine-tune a pre-trained TTS model to improve speech quality and adapt to specific speaker characteristics.

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Approach](#approach)
- [Performance Evaluation](#performance-evaluation)
- [Audio Samples](#audio-samples)
- [Installation](#installation)

## Introduction
This is a Microsoft SpeechT5 model fine-tuned for Hindi.

## Dataset Description
- **Source**: 
  - [English Technical Interview Dataset](https://www.kaggle.com/datasets/shigrafs/english-technical-interview)
  - [English WAV Dataset](https://www.kaggle.com/datasets/shigrafs/english-wav)
- **Content**:
  - Number of audio files: 49,756
  - Number of speakers: 1
  - Types of speech: Scripted
- **Preprocessing Steps**:
  - Normalization of audio levels
  - Segmentation into shorter clips
  - Tokenization of text data

## Model Architecture
This project utilizes the Microsoft Text-To-Speech architecture, which is based on the Google T5 model.

### Key Features
- [List any specific features or modifications made to the model]

## Approach
The dataset stored on Kaggle contains the WAV files and labels, which are loaded into a dataset dictionary along with their array representation and sampling rate. All characters are extracted, and new characters found in the data are added to the tokenizer. The Speechbrain model is used to generate speaker embeddings to help the model adjust its output.

The dataset is prepared, and longer files are trimmed to meet the model requirements (SpeechT5 takes 600 token length of input). The dataset is split into training and testing sets, and a data collator is implemented to handle padding. Training arguments are set up along with a remote repository. The model is trained for 7 epochs in steps to accommodate limited Colab GPU access, and then pushed to a HuggingFace repository.

## Training Setup
- **Environment**:
  - Python version: 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0]
- **Libraries**:
  - PyTorch: 2.5.0+cu121
  - NumPy: 1.26.4
  - Pandas: 2.2.2
  - Matplotlib: 3.7.1
- **Hyperparameters**:
  - Learning rate: [Specify Value]
  - Batch size: [Specify Value]
  - Number of epochs: 20

### Training Logs
**Loss Curve**:  
Include a description of the loss curve and what it indicates about training progress. Ensure CUDA is enabled locally to use GPU resources.

## Performance Evaluation
- **Metrics**:
  - Mean Opinion Score (MOS): [Specify Score]
  - Character Error Rate (CER): [Specify Score]
- **Results**:
  - Pre-trained model: [Specify Metrics]
  - Fine-tuned model: [Specify Metrics]

**Analysis**: Describe performance improvements observed during fine-tuning.

## Audio Samples
Listen to samples generated by both the pre-trained and fine-tuned models:

- **Pre-trained Model**: [Link or description of the audio sample]
- **Fine-tuned Model**: [Link or description of the audio sample]

## Installation
To set up the environment, run the following commands:

```bash
git clone https://github.com/MohammadBinAftab/TTS_ENGLISH.git
cd TTS_ENGLISH
pip install -r requirements.txt
