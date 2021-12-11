# EECS595_Final_Project
This is code for our EECS595_Final_Project: Dual Multi-head Co-Attention For Abstract Meaning Reading Comprehension. We work on SemEval-2021 shared task 4, which requires the participating system to fill in the the correct answer from five candidates of abstract concepts in a cloze-style to replace the @Placeholder in the question. We apply DUal Multi-head Co-Attention (DUMA) model on top of ELECTRA encoder to classify the options. As a result, we get 89.95% for subtask1, 91.41% for subtask2 and 90.59% for subtask3.

## Dataset
Go to https://competitions.codalab.org/competitions/26153#learn_the_details-overview to learn more about SemEval-2021 shared task 4. The data and baseline code are available at https://github.com/boyuanzheng010/SemEval2021-Reading-Comprehension-of-Abstract-Meaning. 

## Environment Setup
This project requires pytorch-lightning, transformers and datasets (please use requirements.txt for installation)
```
pip install -r requirements.txt
```

## Run
We try 3 different methods: Electra Pretrained model, Electra + MAMC(Multi-head Attention Multichoice Classification) and Electra + DUal Multi-head Co-Attention (DUMA). There are 3 subtasks so we get 9 .py files. You can run the corresponding code you interested. For example, To get our result for task1 using Electra Pretrained model:
```
python Electra_task1.py
```
