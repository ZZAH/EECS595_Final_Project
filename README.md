# EECS595_Final_Project
This is code for our EECS595_Final_Project: Dual Multi-head Co-Attention For Abstract Meaning Reading Comprehension. We work on SemEval-2021 shared task 4, which requires the participating system to fill in the the correct answer from five candidates of abstract concepts in a cloze-style to replace the @Placeholder in the question. In this report, we basically choose ELECTRA as our encoder, and try to add Multihead Attention Multichoice Classifier(MAMC) and DUal Multi-head Co-Attention (DUMA) classifier as a one-layer classifier. They both achieve higher performance than ELECTRA itself. On conclusion, our ELECTRA + DUMA approach tends to perform out other methods as our best result, it ranks 3rd for task 1 and 5th for task 2 with the accuracy of 89.95\%, 91.41\%.

## Dataset
Go to https://competitions.codalab.org/competitions/26153#learn_the_details-overview to learn more about SemEval-2021 shared task 4. The data and baseline code are available at https://github.com/boyuanzheng010/SemEval2021-Reading-Comprehension-of-Abstract-Meaning. The data are stored in JSONL format. Copy the data folder to your pwd.

## Environment Setup
This project requires pytorch-lightning, transformers and datasets (please use requirements.txt for installation)
```
pip install -r requirements.txt
```

## Run python file
We try 3 different methods: Electra Pretrained model, Electra + MAMC(Multi-head Attention Multichoice Classification) and Electra + DUal Multi-head Co-Attention (DUMA). You can run the corresponding code you interested. 

For example, To get our result for task1-task3 using Electra Pretrained model:
```
# subtask1
python Electra.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
To get our result for task1-task3 using Electra + MAMC model:
```
# subtask1
python Electra_MAMC.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra_MAMC.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra_MAMC.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra_MAMC.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
To get our result for task1-task3 using Electra + DUMA model:
```
# subtask1
python Electra_DUMA.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra_DUMA.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra_DUMA.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra_DUMA.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
If you are interested in the result of Roberta model, you can run Roberta.py like others. But the performance of Roberta is not so good.


## Run google colab files
We do all experiments in google colab. All files are under Google Colab NLP folders. You can download and open with google colab. You'd better upgrade to Colab Pro+ to run the code quicklly. The resulting screenshots are under Image folder.
