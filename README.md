# Refresh: Ranking Sentences for Extractive Summarization with Reinforcement Learning

This repository releases our code for the Refresh model. It is improved from our code for [Sidenet](https://github.com/shashiongithub/sidenet). It uses Tensorflow 0.10, please use scripts provided by Tensorflow to translate them to newer upgrades. 

Please contact me at shashi.narayan@gmail.com for any question.

Please cite this paper if you use our code or data:

**Ranking Sentences for Extractive Summarization with Reinforcement Learning, Shashi Narayan, Shay B. Cohen and Mirella Lapata, NAACL 2018.**

> Single document summarization is the task of producing a shorter version of a document while preserving its principal information content. In this paper we conceptualize extractive summarization as a sentence ranking task and propose a novel training algorithm which globally optimizes the ROUGE evaluation metric through a reinforcement learning objective. We use our algorithm to train a neural summarization model on the CNN and DailyMail datasets and demonstrate experimentally that it outperforms state-of-the-art extractive and abstractive systems when evaluated automatically and by humans.

## CNN and Dailymail Data

In addition to our code, please find links to additional files which are not uploaded here. 

#### Preprocessed Data and Word Embedding File

* [Pretrained word embeddings](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-1-billion-benchmark-wordembeddings.tar.gz) trained on "1 billion word language modeling benchmark r13output" (405MB)
* [Preprocessed CNN and DailyMail data](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-preprocessed-input-data.tar.gz): Articles are tokenized/segmented with the original case. Then, words are replaced with word ids in the word embedding file with (PAD_ID = 0, UNK_ID = 1). (1.9GB) 
* [Original Test and Validation mainbody data](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-CNN-DM-Filtered-TokenizedSegmented.tar.gz): These files are used to assemble summaries. (35MB)
* [Gold Test and Validation highlights](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-baseline-gold-data.tar.gz): These files are used to estimate ROUGE scores. (11MB)

#### Best Pretrained Models

We train for a certain number of epochs and then we estimate ROUGE score on the validation set after each epoch. The chosen models are the best ones performing on the validation set.  

* [CNN and DailyMail Pretrained Models](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-pretrained-models.tar.gz) (1.8GB)

#### Human Evaluation Data

We have selected 20 (10 CNN and 10 DailyMail) articles. Please see our paper for the experiment setup.

* [CNN and DailyMail Human Evaluation Data](http://kinloch.inf.ed.ac.uk/public/Refresh-NAACL18-human-evaluations.tar.gz)

## Training and Evaluation Instructions

Please download data using the above links and then either update `my_flags.py` for the following parameters or pass them as in-line arguments:

```
pretrained_wordembedding: /address/data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec (Pretrained wordembedding file trained on the one million benchmark data)
preprocessed_data_directory: /address/data/preprocessed-input-directory (Preprocessed news articles)
gold_summary_directory: /address/data/Baseline-Gold-Models (Gold summary directory)
doc_sentence_directory: /address/data/CNN-DM-Filtered-TokenizedSegmented (Directory where document sentences are kept)
```

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
```

## Oracle Estimation

Check our "scripts/oracle-estimator" to compute multiple oracles for your own dataset for training. 

## Blog post and Live Demo

You could find a live demo of Refresh [here](http://kinloch.inf.ed.ac.uk/sidenet.html).

See [here](https://nurture.ai/p/e5c2a653-404a-4af8-b35f-e9e0d17fd272) for a light introduction of our paper written by [nurture.ai](https://nurture.ai).

