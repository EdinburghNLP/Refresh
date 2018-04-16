# Refresh: Ranking Sentences for Extractive Summarization with Reinforcement Learning

This repository releases codes for our Refresh model, an improved version of [Sidenet](https://github.com/shashiongithub/sidenet). They use Tensorflow 0.10, please use scripts provided by Tensorflow to translate them to newer upgrades. 

Please contact me at shashi.narayan@ed.ac.uk for any question.

Please cite this paper if you use any of these:

**Ranking Sentences for Extractive Summarization with Reinforcement Learning, Shashi Narayan, Shay B. Cohen and Mirella Lapata, NAACL 2018.**

> Single document summarization is the task of producing a shorter version of a document while preserving its principal information content. In this paper we conceptualize extractive summarization as a sentence ranking task and propose a novel training algorithm which globally optimizes the ROUGE evaluation metric through a reinforcement learning objective. We use our algorithm to train a neural summarization model on the CNN and DailyMail datasets and demonstrate experimentally that it outperforms state-of-the-art extractive and abstractive systems when evaluated automatically and by humans.

## CNN and Dailymail Data

#### Preprocessed Data and Word Embedding File

#### Pretrained Model and Outputs

#### Human Evaluation Data


## Training and Evaluation Instructions

We run for certain number of epochs and then we estimate ROUGE scores and 

#### CNN 

```
mkdir -p /address/to/training/directory/cnn-reinforcementlearn-singlesample-from-moracle-noatt-sample5

# Training
python document_summarizer_training_testing.py --use_gpu /gpu:2 --data_mode cnn --train_dir /address/to/training/directory/cnn-reinforcementlearn-singlesample-from-moracle-noatt-sample5 --num_sample_rollout 5 > /address/to/training/directory/cnn-reinforcementlearn-singlesample-from-moracle-noatt-sample5/train.log

# Evaluation
python document_summarizer_training_testing.py --use_gpu /gpu:2 --data_mode cnn --exp_mode test --model_to_load 11 --train_dir /address/to/training/directory/cnn-reinforcementlearn-singlesample-from-moracle-noatt-sample5 --num_sample_rollout 5 > /address/to/training/directory/cnn-reinforcementlearn-singlesample-from-moracle-noatt-sample5/test.model11.log
```

#### DailyMail

```
mkdir -p /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15

# Training
python document_summarizer_training_testing.py --use_gpu /gpu:2 --data_mode dailymail --train_dir /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15 --num_sample_rollout 15 > /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15/train.log

# Evaluation
python document_summarizer_training_testing.py --use_gpu /gpu:2 --data_mode dailymail --exp_mode test --model_to_load 7 --train_dir /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15 --num_sample_rollout 15 > /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15/test.model7.log

python document_summarizer_training_testing.py --use_gpu /gpu:2 --data_mode dailymail --exp_mode test --model_to_load 10 --train_dir /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15 --num_sample_rollout 15 > /address/to/training/directory/dailymail-reinforcementlearn-singlesample-from-moracle-noatt-sample15/test.model10.log
```

## Blog and Live Demo

You could find a live demo of Refresh [here](http://kinloch.inf.ed.ac.uk/sidenet.html). 

[nurture.ai](https://nurture.ai) has written a [blog](https://nurture.ai/p/e5c2a653-404a-4af8-b35f-e9e0d17fd272) on our paper.




CNN and DailyMail model
Wordembedding file
Original and Preprocessed Training Data
Model Outputs
Human Evaluation Data
Code

Preprocessing codes: estimate multioracle

