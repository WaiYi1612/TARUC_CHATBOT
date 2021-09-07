from chatterbot import ChatBot, conversation
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import os #for open link


# Creating ChatBot Instance
chatbot = ChatBot(
    'TARUCBot',
    #clean up the receive of input whitespace may add up other preprocessors in future
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace'
    ],
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90,
           
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 

# Training with Personal Ques & Ans 
#training_data_quesans = open('training_data/ques_ans.txt').read().splitlines()
#training_data_personal = open('training_data/Introduce.txt').read().splitlines()

#training_data = training_data_quesans + training_data_personal

#Introduce + Programme + Financial
training_introduce = open('training_data/Introduce.txt',encoding='utf-8', errors='ignore').read().splitlines()
training_programme = open('training_data/Programme.txt',encoding='utf-8', errors='ignore').read().splitlines()
training_financial_aid = open('training_data/FinancialAid.txt',encoding='utf-8', errors='ignore').read().splitlines()
training_short_form = open('training_data/ShortForm.txt',encoding='utf-8', errors='ignore').read().splitlines()
training_chinese_dialogue = open('training_data/ChineseDialogue.txt',encoding='utf-8', errors='ignore').read().splitlines()

conversation = training_introduce + training_programme + training_financial_aid + training_short_form + training_chinese_dialogue

trainer = ListTrainer(chatbot)
trainer.train(conversation)  


# Training with English Corpus Data 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train(
    'chatterbot.corpus.english',
    'chatterbot.corpus.chinese'
) 


