import json,os,sys,re
from naiveBayesClassifier import tokenizer
from naiveBayesClassifier.trainer import Trainer
from naiveBayesClassifier.classifier import Classifier
##IMPORTS

'''
Usage:

    python GuessDisease.py "symptomA symptomB symptomC"
Example INPUT:
    python GuessDisease.py "agitation exhaustion vomit"
Example OUTPUT:

    {
    "disease": "influenza"
    }


'''



##SETTING UP
diseaseclassifier = Trainer(tokenizer) #STARTS CLASIFIERS
with open("Dataset.csv", "r") as file: #OPENS DATASET
    for i in file: #FOR EACH LINE
       lines = file.next().split(",") #PARSE CSV <DISEASE> <SYMPTOM>
       diseaseclassifier.train(lines[1],  lines[0]) #TRAINING
diseaseclassifier = Classifier(diseaseclassifier.data, tokenizer)
classification = diseaseclassifier.classify(sys.argv[1]) #CLASIFY INPUT
print classification[0] #PRINT CLASIFICATION
