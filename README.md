# EECS595_Final_Project
This is code for our EECS595_Final_Project: Dual Multi-head Co-Attention For Abstract Meaning Reading Comprehension. We work on SemEval-2021 shared task 4, which requires the participating system to fill in the the correct answer from five candidates of abstract concepts in a cloze-style to replace the @Placeholder in the question. We apply DUal Multi-head Co-Attention (DUMA) model on top of ELECTRA encoder to classify the options. As a result, we get 89.95% for subtask1, 91.41% for subtask2 and 90.59% for subtask3.

## Dataset
Go to https://competitions.codalab.org/competitions/26153#learn_the_details-overview to learn more about SemEval-2021 shared task 4. The data and baseline code are available at https://github.com/boyuanzheng010/SemEval2021-Reading-Comprehension-of-Abstract-Meaning. The data are stored in JSONL format.

## Environment Setup
This project requires pytorch-lightning, transformers and datasets (please use requirements.txt for installation)
```
pip install -r requirements.txt
```

## Run
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

