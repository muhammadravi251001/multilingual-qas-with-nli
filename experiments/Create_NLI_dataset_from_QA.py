#!/usr/bin/env python
# coding: utf-8

# # Define tool and model of the tool

# In[209]:


get_ipython().system('nvidia-smi')


# Below, it is some settings to run in my local.

# In[210]:


import os, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# You can tweak your settings too in code below.

# In[211]:


import sys

NAME = "idk-mrc"
NO_ANSWER_STATEMENT = "Tidak ada jawaban"

TASK_NER_NAME = "ner"
MODEL_NER_NAME = "ageng-anugrah/indobert-large-p2-finetuned-ner"

TASK_CHUNKING_NAME = "token-classification"
MODEL_CHUNKING_NAME = "ageng-anugrah/indobert-large-p2-finetuned-chunking"

MODEL_SIMILARITY_NAME = "paraphrase-multilingual-mpnet-base-v2"
URL_STOPWORD = "https://raw.githubusercontent.com/6/stopwords-json/master/stopwords-all.json"

TASK_PARAPHRASER_NAME = "text2text-generation"
MODEL_PARAPHRASER_NAME = ""

# Uncomment sys.maxsize to create all of the data, 
# else if you want to debugging

# SAMPLE = sys.maxsize
SAMPLE = 250


# # Import anything

# In[212]:


import transformers
import evaluate
import torch
import operator
import re
import sys
import collections
import string
import contextlib
import gc
import random
import string
import requests

import numpy as np
import pandas as pd
import torch.nn as nn

from multiprocessing import cpu_count
from evaluate import load
from nusacrowd import NusantaraConfigHelper
from datetime import datetime
from huggingface_hub import notebook_login
from tqdm import tqdm
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, util

from datasets import (
    load_dataset, 
    Dataset,
    DatasetDict
)
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback, 
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    pipeline
)


# # Retrieve QA dataset

# In[213]:


print("PROGRAM STARTED")


# In[214]:


conhelps = NusantaraConfigHelper()
data_qas = conhelps.filtered(lambda x: 'idk_mrc' in x.dataset_name)[0].load_dataset()

df_train = pd.DataFrame(data_qas['train'])
df_validation = pd.DataFrame(data_qas['validation'])
df_test = pd.DataFrame(data_qas['test'])

cols = ['context', 'question', 'answer']
new_df_train = pd.DataFrame(columns=cols)

for i in tqdm(range(len(df_train['context']))):
    for j in df_train["qas"][i]:
        if len(j['answers']) != 0:
            new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                                'question': j['question'], 
                                                'answer': {"text": j['answers'][0]['text'], 
                                                           "answer_start": j['answers'][0]['answer_start'], 
                                                           "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                           ignore_index=True)
        else:
            new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                                'question': j['question'], 
                                                'answer': {"text": str(), 
                                                           "answer_start": 0, 
                                                           "answer_end": 0}}, 
                                                           ignore_index=True)

cols = ['context', 'question', 'answer']
new_df_val = pd.DataFrame(columns=cols)

for i in tqdm(range(len(df_validation['context']))):
    for j in df_validation["qas"][i]:
        if len(j['answers']) != 0:
            new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                            'question': j['question'], 
                                            'answer': {"text": j['answers'][0]['text'], 
                                                       "answer_start": j['answers'][0]['answer_start'], 
                                                       "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                       ignore_index=True)
        else:
            new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                            'question': j['question'], 
                                            'answer': {"text": str(), 
                                                       "answer_start": 0, 
                                                       "answer_end": 0}}, 
                                                       ignore_index=True)        

cols = ['context', 'question', 'answer']
new_df_test = pd.DataFrame(columns=cols)

for i in tqdm(range(len(df_test['context']))):
    for j in df_test["qas"][i]:
        if len(j['answers']) != 0:
            new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                            'question': j['question'], 
                                            'answer': {"text": j['answers'][0]['text'], 
                                                       "answer_start": j['answers'][0]['answer_start'], 
                                                       "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                       ignore_index=True)
        else:
            new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                            'question': j['question'], 
                                            'answer': {"text": str(), 
                                                       "answer_start": 0, 
                                                       "answer_end": 0}}, 
                                                       ignore_index=True)

train_dataset = Dataset.from_dict(new_df_train)
validation_dataset = Dataset.from_dict(new_df_val)
test_dataset = Dataset.from_dict(new_df_test)

data_qas = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})
data_qas


# # Convert to NLI, with hypothesis being just do concat question & answer

# ## Convert Dataset to DataFrame format

# In[215]:


# 42, the answer to life the universe and everything

seed_value = 42
random.seed(seed_value)


# In[216]:


# If you want to training all of the data (prod),
# this code will convert to DataFrame.
# However, if you want to debug (not-prod),
# this code will convert SAMPLE of your DataFrame

if SAMPLE == sys.maxsize:
    data_qas_train_df = pd.DataFrame(data_qas["train"][:SAMPLE])
    data_qas_val_df = pd.DataFrame(data_qas["validation"][:SAMPLE])
    data_qas_test_df = pd.DataFrame(data_qas["test"][:SAMPLE])

else:
    data_qas_train_df = (pd.DataFrame(data_qas["train"])).sample(n=SAMPLE, random_state=seed_value)
    data_qas_val_df = (pd.DataFrame(data_qas["validation"])).sample(n=SAMPLE, random_state=seed_value)
    data_qas_test_df = (pd.DataFrame(data_qas["test"])).sample(n=SAMPLE, random_state=seed_value)

    data_qas_train_df = data_qas_train_df.reset_index(drop=True)
    data_qas_val_df = data_qas_val_df.reset_index(drop=True)
    data_qas_test_df = data_qas_test_df.reset_index(drop=True)


# ## Retrieve answer text only

# In[217]:


# Only retrieve answer text
# Because, we do not use answer_start
# and answer_end

def retrieve_answer_text(data):
    for i in range(len(data)):
        data['answer'][i] = data['answer'][i]['text']
    return data


# In[218]:


data_qas_train_df = retrieve_answer_text(data_qas_train_df)
data_qas_val_df = retrieve_answer_text(data_qas_val_df)
data_qas_test_df = retrieve_answer_text(data_qas_test_df)


# ## Create NLI dataset from copy of QA dataset above

# In[219]:


data_nli_train_df = data_qas_train_df.copy()
data_nli_val_df = data_qas_val_df.copy()
data_nli_test_df = data_qas_test_df.copy()


# In[220]:


data_qas_train_df


# In[1570]:


data = {
    'context': ["Baudouin adalah putra Eustace II, Comte Boulogne dan Ide dari Lorraine (putri Godefroy III, Adipati Lorraine Hilir), dan adik Eustace III, Comte Boulogne dan Godefroy dari Bouillon. Sebagai saudara bungsu, Baudouin awalnya ditujukan untuk berkarier di gereja, namun ia menyerah pada sekitar tahun 1080; menurut Willelmus dari Tirus, yang hidup pada abad ke-12 dan tidak mengenal Baudouin secara pribadi: \"di masa mudanya, Baudouin juga memelihara seni liberal. Ia menjadi seorang ulama, dikatakan, dan karena keturunannya yang terkenal, memiliki manfaat yang biasa disebut prebend di gereja-gereja di Reims, Cambrai, dan Liège.\" Setelah itu ia tinggal di Normandia, di mana ia menikahi Godehilde (atau Godvera) de Toeni, putri Raoul de Conches dari keluarga bangsawan Anglo-Norman (dan sebelumnya bertunangan dengan istri Robert de Beaumont). Ia kembali ke Lorraine Hilir dalam rangka untuk mengendalikan kadipaten Verdun (yang sebelumnya dipegang oleh Godefroy)."],
    'question': ["Siapa ibu Baudouin I ?"],
    'answer': ["Ide dari Lorraine"]
}


# In[1571]:


#data_debug = pd.DataFrame(data)
#data_debug


# ## Convert context pair to premise (only renaming column)

# In[1572]:


# Renaming it, just for consistency

data_nli_train_df = data_nli_train_df.rename(columns={"context": "premise"})
data_nli_val_df = data_nli_val_df.rename(columns={"context": "premise"})
data_nli_test_df = data_nli_test_df.rename(columns={"context": "premise"})


# In[1573]:


#data_debug = data_debug.rename(columns={"context": "premise"})


# # Add contradiction label cases

# ## Import pipeline to create contradiction cases

# In[1574]:


nlp_tools_ner = pipeline(task = TASK_NER_NAME, 
                     model = MODEL_NER_NAME, 
                     tokenizer = AutoTokenizer.from_pretrained(MODEL_NER_NAME, 
                                                               model_max_length=512, 
                                                               truncation=True),
                     aggregation_strategy = 'simple')


# In[1575]:


nlp_tools_chunking = pipeline(task = TASK_CHUNKING_NAME, 
                     model = MODEL_CHUNKING_NAME, 
                     tokenizer = AutoTokenizer.from_pretrained(MODEL_CHUNKING_NAME, 
                                                               model_max_length=512, 
                                                               truncation=True),
                     aggregation_strategy = 'simple')


# ## Add NER and chunking tag column in DataFrame

# In[1576]:


# This code useful for cleaning the data (text)

def remove_space_after_number_and_punctuation(text):
    pattern = r'(\d+)\s*([.,])\s*(?=\S|$)'
    cleaned_text = re.sub(pattern, r'\1\2', text)
    return cleaned_text


# In[1577]:


# This code useful for tagging the entire premise
# with NER and chunking tools

def add_premise_tag(data, tag, index, premise_array, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

    if tag == "ner": tools=ner
    else: tools=chunking
    
    # If the tools detected nothing, retrieve NO TOKEN DETECTED
    if len(tools(data['premise'][index])) == 0:
        premise_array.append("NO TOKEN DETECTED")
    
    # Else if, the tools detected something, retrieve all of the entity and the word associated
    else:
        for j in tools(data['premise'][index]):
            tag_premise = (j['entity_group'], remove_space_after_number_and_punctuation(j['word']))
            premise_array.append(tag_premise)

    return premise_array


# In[1578]:


# Function for clean the text off punctuation

def remove_punctuation(text):
    cleaned_text = text.lstrip(string.punctuation)
    cleaned_text = cleaned_text.rstrip(string.punctuation)
    return cleaned_text


# In[1579]:


# This code useful for tagging the entire answer
# with NER and chunking tools

def add_answer_tag(answer, tag, premise_array, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

    if tag == "ner": tools=ner
    else: tools=chunking

    tag_answer_list = list()
    
    if tag == "ner":
        
        # If tools in premise detecting some token
        
        if len(premise_array) != 0:
            
            for i in premise_array:
                
                # Extract the label and the word of premise
                
                label_from_premise_tag = i[0]
                word_from_premise_tag = remove_space_after_number_and_punctuation(i[1])

                # With assumption, that I do not dividing label when
                # there is more than one label in one word answer.
                # Instead, I give a NULL.

                if word_from_premise_tag.lower() == answer.lower():
                    tag_answer = (label_from_premise_tag, word_from_premise_tag)
                    break

                # Or, I could do this: to reducing NULL label 
                # with subset of string not really with the entire string.
                
                elif answer.lower() in word_from_premise_tag or word_from_premise_tag in answer.lower():
                    tag_answer = (label_from_premise_tag, answer.lower())
                    break
                
                # Then, if you still do not find the word, NULL given
                
                else:
                    tag_answer = ("NULL", answer)
            
            tag_answer_list.append(tag_answer)

        # If tools in premise NOT detecting some token, NULL given
        
        else:
            tag_answer = ("NULL", answer)
            tag_answer_list.append(tag_answer)
    
    elif tag == "chunking":
        
        # In chunking, it's slightly different because 
        # the basic assumption is that there are no NULL chunks,
        # so it will capture all the chunk labels.
        
        retrieved_from_tools = tools(answer)

        if len(retrieved_from_tools) != 0:
            
            for i in retrieved_from_tools:
                tag_answer = (i['entity_group'], i['word'])
                tag_answer_list.append(tag_answer)
        
        # So, it rarely going down there
        # But, if really going down there
        # from basic assumption, there is no NULL in chunking
        # We can check subset of the sentence from premise,
        # if found, we can take that particular label
        
        else:
            
            for i in premise_array:
                
                # Extract the label and the word of premise
                
                label_from_premise_tag = i[0]
                word_from_premise_tag = remove_space_after_number_and_punctuation(i[1])
                
                # Take label from subset of sentence from premise
                
                if answer.lower() in word_from_premise_tag:
                    tag_answer = (label_from_premise_tag, answer.lower())
                    tag_answer_list.append(tag_answer)
                    break
            
            # Use for and then direct else (for-else),
            # if for-loop above not getting the break statement
            
            else:
                tag_answer = ("NULL", answer)
                tag_answer_list.append(tag_answer)
        
    return tag_answer_list


# In[1580]:


# This is a helper code to run
# process for add tag to entire premise
# and answer

def add_ner_and_chunking_all_tag(data):
    
    data['ner_tag_answer'] = ""
    data['chunking_tag_answer'] = ""
    
    data['ner_tag_premise'] = ""
    data['chunking_tag_premise'] = ""
    
    for i in tqdm(range(len(data))):
        
        answer = data['answer'][i]
        premise = data['premise'][i]
        
        ner_premise_array = list()
        chunking_premise_array = list()
                                                
        data['ner_tag_premise'][i] = add_premise_tag(data, "ner", i, ner_premise_array)
        data['chunking_tag_premise'][i] = add_premise_tag(data, "chunking", i, chunking_premise_array)
        
        data['ner_tag_answer'][i] = add_answer_tag(answer, "ner", data['ner_tag_premise'][i])
        data['chunking_tag_answer'][i] = add_answer_tag(answer, "chunking", data['chunking_tag_premise'][i])
    
    return data


# In[1581]:


data_nli_train_df = add_ner_and_chunking_all_tag(data_nli_train_df)
data_nli_val_df = add_ner_and_chunking_all_tag(data_nli_val_df)
data_nli_test_df = add_ner_and_chunking_all_tag(data_nli_test_df)


# In[1582]:


#data_debug = add_ner_and_chunking_all_tag(data_debug)
#data_debug


# # Create wrong answer

# In[1583]:


# This function useful for sorting the closest distance
# by using embedding

model_similarity = SentenceTransformer(MODEL_SIMILARITY_NAME)

def return_similarity_sorted_array(right_answer, sentence_array, model=model_similarity):
    
    right_answer = right_answer.lower()
    
    embedding_right_answer = model.encode([right_answer], convert_to_tensor=True, device=device)
    embedding_sentence_array = model.encode(sentence_array, convert_to_tensor=True, device=device)
    
    # Using cosine scores to calculate
    cosine_scores = util.pytorch_cos_sim(embedding_right_answer, embedding_sentence_array)
    
    sorted_indices = cosine_scores.argsort(descending=True)[0]
    sorted_array = [sentence_array[i] for i in sorted_indices]
    
    return sorted_array


# In[1584]:


# This function useful for
# removing value with hash.
# Because, from label-tagging before
# Some data have a hash symbol, because
# that data was part of a word fragment

def remove_values_with_hash(arr):
    return [item for item in arr if "#" not in item]


# In[1585]:


# Retrieve stopword from all language

response = requests.get(URL_STOPWORD)

if response.status_code == 200:
    stopword_data = response.json()
else:
    print("Failed to download stopword JSON.")

stopword_data = set([item for sublist in list(stopword_data.values()) for item in sublist])


# In[1586]:


# This function just retrieve random word
# of entire premise

def select_random_word(text, answer, stopword_data=stopword_data):

    words = re.findall(r'\w+', text.lower())
    
    # Filtering to remove stopword and punctuation
    filtered_words = [word for word in words if word not in stopword_data and word not in string.punctuation]
    
    # If filtered words less than answer
    # only take one word as random word
    
    if len(filtered_words) < len(answer.split()):
        random_word = random.choice(filtered_words)
    
    # But, if filtered words NOT less than answer
    # take a same length word as a random word
    # with the same order as filtered words
    
    else:
        start_index = random.randint(0, len(filtered_words) - len(answer.split()))
        random_word_array = filtered_words[start_index : start_index + len(answer.split())]
        random_word = ' '.join(random_word_array)
    
    return random_word.strip()


# In[1587]:


# This function useful for find the same order
# of sequence, this function will used in
# chunking domain, to classify whether an
# answer is word or a sentence

def find_order(premise, answer):
    
    results = []
    answer_labels = [item[0] for item in answer]
    
    i = 0
    
    while i < len(premise):
        if premise[i][0] == answer_labels[0]:
            j = 0
            matching_words = []
            
            while i + j < len(premise) and j < len(answer_labels) and premise[i + j][0] == answer_labels[j]:
                matching_words.append(premise[i + j][1])
                j += 1
            
            if j == len(answer_labels):
                results.append(" ".join(matching_words))
            
        i += 1
    
    return results


# In[1588]:


# This function useful for grouping same tag-label 
# between answer and word (or sentence) in an entire premise

def grouping_same_tag(tag_answers, tag_premises, same_tag_array, tag):
    
    # In NER, basicly you need to iterate
    # to find a same tag-label
    
    if tag == "ner":
        
        for tag_premise in tag_premises:

            label_tag_premise = tag_premise[0]
            word_premise = tag_premise[1]

            for tag_answer in tag_answers:

                label_tag_answer = tag_answer[0]

                if label_tag_answer == label_tag_premise:
                    same_tag_array.append(word_premise)
                    
    # In Chunking, slightly different
    # you need to find subset for find the same order
    # of sequence with find_order() function
    
    elif tag == "chunking":
        
        matching_words = find_order(tag_premises, tag_answers)
        
        # If there is a correct order of subset answer to premise, add it word
        
        if len(matching_words) != 0:
            for word in matching_words:
                same_tag_array.append(word)
        
        # If no matching words, use NER algorithm above
        
        else:
            grouping_same_tag(tag_answers, tag_premises, same_tag_array, "ner")

    # Still, filter value with hash
    
    return remove_values_with_hash(same_tag_array)


# In[1589]:


# This function useful for
# checking text if only
# contain punctuation, no words at all 

def contains_only_punctuation(text):
    return all(char in string.punctuation for char in text)


# In[1590]:


# This function useful for
# filter overlapping right answer and wrong answer
# that provided in plausible answer

def filtering_plausible_answer(answer, plausible_answer_array):
    
    if type(plausible_answer_array) == str: 
        plausible_answer_array = list([plausible_answer_array])
    
    answer = answer.lower()
    
    plausible_answer_array = [item.lower().strip() for item in plausible_answer_array]
    plausible_answer_array = [string for string in plausible_answer_array if not contains_only_punctuation(string)]
    plausible_answer_array = [remove_punctuation(text) for text in plausible_answer_array]
    
    final_plausible_answer_array = list()
    answer_words = set(remove_punctuation(text) for text in answer.split())
    
    # For check overlapping answer, using set of word,
    # and so, check for intersection
    
    for plausible_answer in plausible_answer_array:
        plausible_answer_words = set(plausible_answer.split())
        if not plausible_answer_words.intersection(answer_words):
            if not all(word in answer for word in plausible_answer.split()):
                final_plausible_answer_array.append(plausible_answer)
    
    return final_plausible_answer_array


# In[1591]:


# This function useful for
# detecting number, date, time
# to give plausible answer more
# "make sense" answer

def is_number(input_str):
    pattern = r'^.*\b\d+(\.\d+)?\b.*$'
    return re.match(pattern, input_str) is not None

def is_date(input_str):
    pattern_1 = r'\d{1,2} [A-Za-z]+(?: \d{4})?'
    pattern_2 = r'^\d{4}-\d{2}-\d{2}$'
    return (re.match(pattern_1, input_str) or re.match(pattern_2, input_str)) is not None

def is_time(input_str):
    pattern = r'^\d{2}:\d{2}:\d{2}$'
    return re.match(pattern, input_str) is not None

def check_regex(right_answer, plausible_answer_array):
    
    if is_number(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_number(item)]
    
    elif is_date(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_date(item)]
    
    elif is_time(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_time(item)]
        
    else:
        plausible_answer_array = [item for item in plausible_answer_array if (not is_number(item) or 
                                                                              not is_date(item) or 
                                                                              not is_time(item)
                                                                             )]
    
    return plausible_answer_array


# In[1592]:


# This function useful for
# sorting similarity and
# to give final wrong answer 
# and properties

def return_wrong_and_plausible(data, right_answer, index, tag, plausible_answer_array, premise,
                       NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):

    if tag == "ner": slice = 'same_ner_tag_answer'
    elif tag == "chunking": slice = 'same_chunking_tag_answer'
    else: slice = None

    # Find all the sorted (by similarity) plausible wrong answer, 
    # and remove hask & punctuation only answer
    
    if slice != None:
        wrong_answer_array = return_similarity_sorted_array(right_answer, data[slice][index])
    
    else:
        wrong_answer_array = return_similarity_sorted_array(right_answer, plausible_answer_array)
    
    # Below, do the filtering to plausible answer
    
    plausible_answer_array = remove_values_with_hash(wrong_answer_array)
    plausible_answer_array = filtering_plausible_answer(right_answer, plausible_answer_array)
    plausible_answer_array = check_regex(right_answer, plausible_answer_array)
    
    try:
        
        # Only return the most similar to right_answer
        wrong_answer = plausible_answer_array[0].strip()
        
        if tag == "ner": 
            properties = "IDENTICAL NER labels were found, and the highest similarity score same NER array was selected"
        
        elif tag == "chunking":
            properties = "IDENTICAL Chunking labels were found, and the highest similarity score from same Chunking array was selected"
        
        else:
            properties = "NO CHUNKING labels were found, and the highest similarity score from plausible answer was selected"
    
    except:
        
        # Selecting wrong answer from random word in premise
        wrong_answer = select_random_word(premise, right_answer)
        
        # If that random word is overlapping to right answer,
        # iterate again until it is not overlap again
        
        while True:
            
            if filtering_plausible_answer(right_answer, wrong_answer) != []:
                break

            wrong_answer = select_random_word(premise, right_answer)[0]

        # If it still detect overlapped right answer,
        # just assign it with NO_ANSWER_STATEMENT.
        # But, this condition is very-extraordinary
        
        else:
            wrong_answer = NO_ANSWER_STATEMENT
        
        if tag == "ner": 
            properties = "Detected (NER) wrong answer that is the SAME as the right answer, search random word from premise"
        
        elif tag == "chunking":
            properties = "Detected (Chunking) wrong answer that is the SAME as the right answer, search random word from premise"
        
        else:
            properties = "Detected (Random) wrong answer that is the SAME as the right answer, search random word from premise"
    
    # Still need to check/assert the wrong answer
    # and the plausible answer type
    
    assert isinstance(wrong_answer, str)
    assert isinstance(plausible_answer_array, list)
    
    return wrong_answer, plausible_answer_array, properties


# In[1593]:


# This function useful for
# matching regex in the first
# section on create wrong answer
# to detect: number, date, and time

def matching_regex(right_answer, premise):
    
    plausible_answer_array = []

    for word in premise.split():
        
        word = remove_punctuation(word)
        right_answer = remove_punctuation(right_answer)
        
        if (word != right_answer) and (word not in right_answer):
        
            if is_number(word) and is_number(right_answer):
                plausible_answer_array.append(word)

            if is_date(word) and is_date(right_answer):
                plausible_answer_array.append(word)

            if is_time(word) and is_time(right_answer):
                plausible_answer_array.append(word)

    return plausible_answer_array


# In[1594]:


# This function is the main idea to create wrong answer
# Though, this function is helper function

def create_wrong_answer(data, NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):
    
    data['same_ner_tag_answer'] = ""
    data['same_chunking_tag_answer'] = ""
    data['wrong_answer'] = ""
    data['no_answer'] = ""
    data['plausible_answer_based_on_method'] = ""
    data['properties'] = ""
    
    for i in tqdm(range(len(data))):
        
        right_answer = data['answer'][i]
        premise = data['premise'][i]

        same_ner_tag_answer_array = list()
        same_chunking_tag_answer_array = list()

        ner_tag_answer = data['ner_tag_answer'][i]
        ner_tag_premise = data['ner_tag_premise'][i]

        chunking_tag_answer = data['chunking_tag_answer'][i]
        chunking_tag_premise = data['chunking_tag_premise'][i]
        
        # If that row of data is unanswerable, do this, then continue
        
        if right_answer == "":
            data['properties'][i] = "Unanswerable question"
            data['wrong_answer'][i] = "NULL"
            data['no_answer'][i] = "NULL"
            data['plausible_answer_based_on_method'][i] = "Unanswerable question"
            continue
            
        # Grouped with the same NER & Chunking group, between answer and word of premise
        
        data['same_ner_tag_answer'][i] = grouping_same_tag(ner_tag_answer,
                                                           ner_tag_premise,
                                                           same_ner_tag_answer_array, "ner")
        
        data['same_chunking_tag_answer'][i] = grouping_same_tag(chunking_tag_answer, 
                                                                chunking_tag_premise, 
                                                                same_chunking_tag_answer_array, "chunking")
        
        # Start to create wrong answer
        plausible_answer_array = list()
        
        # Firstly, matching regex
        if is_number(right_answer) or is_date(right_answer) or is_time(right_answer):
            
            plausible_answer_array = matching_regex(right_answer, premise)
            
            if len(plausible_answer_array) != 0:
                plausible_answer_array = return_similarity_sorted_array(right_answer, plausible_answer_array)
                wrong_answer = plausible_answer_array[0].strip()
                data['properties'][i] = "Regex matched with right answer, and get alternative answer"
            
            else:
                wrong_answer = NO_ANSWER_STATEMENT
                data['properties'][i] = "Regex matched with right answer, but no alternative answer"
            
            data['wrong_answer'][i] = wrong_answer
            data['no_answer'][i] = NO_ANSWER_STATEMENT
            data['plausible_answer_based_on_method'][i] = list(set(plausible_answer_array))
            continue

        # Perform NER classification
        # If the NER of the right_answer can be detected, then calculate the distance using semantic 
        # similarity or word vectors between the right_answer and various possible wrong_answers with 
        # the same NER as the right_answer. Once done, proceed to the final wrong_answer.
        
        if data['same_ner_tag_answer'][i] != list():
            wrong_answer, plausible_answer_array, properties = return_wrong_and_plausible(data, right_answer, \
                                                                      i, "ner", plausible_answer_array, premise)
            
        # If the NER of the right_answer cannot be detected (NULL) or context/premise does not contain 
        # any of NER of right_answer, then the POS/Chunking of the right_answer will be identified.
        
        # Perform POS/Chunking classification
        
        else:
            
            # If the POS/Chunking of the right_answer can be detected, then calculate the distance 
            # using semantic similarity or word vectors between the right_answer and various possible 
            # wrong_answers with the same POS/Chunking as the right_answer. Once done, proceed to the 
            # final wrong_answer.
            
            if data['same_chunking_tag_answer'][i] != list():
                wrong_answer, plausible_answer_array, properties = return_wrong_and_plausible(data, right_answer, \
                                                                          i, "chunking", plausible_answer_array, premise)
            
            # If the POS/Chunking of the right_answer cannot be detected (NULL) or context/premise 
            # does not contain any of NER of right_answer, then the final wrong_answer will be chosen 
            # based on a plausible answer.
            
            else:
                for chunking_tag in chunking_tag_premise:
                    plausible_answer_array.append(chunking_tag[1])

                wrong_answer, plausible_answer_array, properties = return_wrong_and_plausible(data, right_answer, \
                                                                          i, "none", plausible_answer_array, premise)
        
        data['properties'][i] = properties
        data['wrong_answer'][i] = wrong_answer
        data['no_answer'][i] = NO_ANSWER_STATEMENT
        data['plausible_answer_based_on_method'][i] = list(set(plausible_answer_array))
            
    return data       


# In[1595]:


data_nli_train_df = create_wrong_answer(data_nli_train_df)
data_nli_val_df = create_wrong_answer(data_nli_val_df)
data_nli_test_df = create_wrong_answer(data_nli_test_df)


# In[1596]:


#data_debug = create_wrong_answer(data_debug)
#data_debug


# In[1597]:


#print("Right answer:", data_debug['answer'][0])
#print()
#print(data_debug['plausible_answer_based_on_method'][0])
#print("Wrong answer:", data_debug['wrong_answer'][0])


# In[1498]:


#1+1


# In[1503]:


#print("Premise:", data_debug['premise'][0])
#print("Question:", data_debug['question'][0])
#print()
#print(data_debug['ner_tag_premise'][0])
#print(data_debug['chunking_tag_premise'][0])


# # Split to two dataset: right dataset & wrong dataset

# In[33]:


# This method is just only
# for aesthetics of column number

def move_to_column_number(data, column_name="hypothesis", column_num=3):

    cols = list(data.columns)
    cols.remove(column_name)
    cols.insert(column_num, column_name)

    data = data[cols]
    
    return data


# In[34]:


# Creating answerable right (entailment label) dataset

columns_to_exclude = ['wrong_answer', 'no_answer']

data_nli_answerable_right_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
data_nli_answerable_right_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
data_nli_answerable_right_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()

data_nli_answerable_right_train_df = data_nli_answerable_right_train_df[data_nli_answerable_right_train_df['answer'] != '']
data_nli_answerable_right_val_df = data_nli_answerable_right_val_df[data_nli_answerable_right_val_df['answer'] != '']
data_nli_answerable_right_test_df = data_nli_answerable_right_test_df[data_nli_answerable_right_test_df['answer'] != '']

data_nli_answerable_right_train_df = data_nli_answerable_right_train_df.reset_index(drop=True)
data_nli_answerable_right_val_df = data_nli_answerable_right_val_df.reset_index(drop=True)
data_nli_answerable_right_test_df = data_nli_answerable_right_test_df.reset_index(drop=True)


# In[35]:


# Creating answerable wrong (contradiction label) dataset

columns_to_exclude = ['answer', 'no_answer']

data_nli_answerable_wrong_train_df = data_nli_train_df[data_nli_train_df['answer'] != '']
data_nli_answerable_wrong_val_df = data_nli_val_df[data_nli_val_df['answer'] != '']
data_nli_answerable_wrong_test_df = data_nli_test_df[data_nli_test_df['answer'] != '']

data_nli_answerable_wrong_train_df = data_nli_answerable_wrong_train_df.drop(columns=columns_to_exclude).copy()
data_nli_answerable_wrong_val_df = data_nli_answerable_wrong_val_df.drop(columns=columns_to_exclude).copy()
data_nli_answerable_wrong_test_df = data_nli_answerable_wrong_test_df.drop(columns=columns_to_exclude).copy()

data_nli_answerable_wrong_train_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)
data_nli_answerable_wrong_val_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)
data_nli_answerable_wrong_test_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)

data_nli_answerable_wrong_train_df = data_nli_answerable_wrong_train_df.reset_index(drop=True)
data_nli_answerable_wrong_val_df = data_nli_answerable_wrong_val_df.reset_index(drop=True)
data_nli_answerable_wrong_test_df = data_nli_answerable_wrong_test_df.reset_index(drop=True)

data_nli_answerable_wrong_train_df = move_to_column_number(data_nli_answerable_wrong_train_df, "answer", 2)
data_nli_answerable_wrong_val_df = move_to_column_number(data_nli_answerable_wrong_val_df, "answer", 2)
data_nli_answerable_wrong_test_df = move_to_column_number(data_nli_answerable_wrong_test_df, "answer", 2)


# In[36]:


# Creating unanswerable right (entailment label) and no-answer dataset

columns_to_exclude = ['wrong_answer', 'no_answer']

data_nli_unanswerable_right_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_right_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_right_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()

data_nli_unanswerable_right_train_df = data_nli_unanswerable_right_train_df[data_nli_unanswerable_right_train_df['answer'] == '']
data_nli_unanswerable_right_val_df = data_nli_unanswerable_right_val_df[data_nli_unanswerable_right_val_df['answer'] == '']
data_nli_unanswerable_right_test_df = data_nli_unanswerable_right_test_df[data_nli_unanswerable_right_test_df['answer'] == '']


# In[37]:


# Creating unanswerable wrong (contradiction label) and no-answer dataset

columns_to_exclude = ['answer', 'wrong_answer']

data_nli_unanswerable_wrong_train_df = data_nli_train_df[data_nli_train_df['answer'] != '']
data_nli_unanswerable_wrong_val_df = data_nli_val_df[data_nli_val_df['answer'] != '']
data_nli_unanswerable_wrong_test_df = data_nli_test_df[data_nli_test_df['answer'] != '']

data_nli_unanswerable_wrong_train_df = data_nli_unanswerable_wrong_train_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_wrong_val_df = data_nli_unanswerable_wrong_val_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_wrong_test_df = data_nli_unanswerable_wrong_test_df.drop(columns=columns_to_exclude).copy()

data_nli_unanswerable_wrong_train_df.rename(columns={'no_answer': 'answer'}, inplace=True)
data_nli_unanswerable_wrong_val_df.rename(columns={'no_answer': 'answer'}, inplace=True)
data_nli_unanswerable_wrong_test_df.rename(columns={'no_answer': 'answer'}, inplace=True)

data_nli_unanswerable_wrong_train_df = move_to_column_number(data_nli_unanswerable_wrong_train_df, "answer", 2)
data_nli_unanswerable_wrong_val_df = move_to_column_number(data_nli_unanswerable_wrong_val_df, "answer", 2)
data_nli_unanswerable_wrong_test_df = move_to_column_number(data_nli_unanswerable_wrong_test_df, "answer", 2)


# In[38]:


# Rather than duplicating the no-answer statement, 
# it's better to remove the excessing row ones.

def balancing_data(data1, data2):
    
    if len(data1) > len(data2):
        data1 = data1.sample(n=len(data2))
    
    elif len(data1) < len(data2):
        data2 = data2.sample(n=len(data1))
        
    return data1, data2


# In[39]:


data_nli_unanswerable_right_train_df, data_nli_unanswerable_wrong_train_df = balancing_data(data_nli_unanswerable_right_train_df,
                                                                                            data_nli_unanswerable_wrong_train_df)

data_nli_unanswerable_right_val_df, data_nli_unanswerable_wrong_val_df = balancing_data(data_nli_unanswerable_right_val_df,
                                                                                        data_nli_unanswerable_wrong_val_df)

data_nli_unanswerable_right_test_df, data_nli_unanswerable_wrong_test_df = balancing_data(data_nli_unanswerable_right_test_df,
                                                                                        data_nli_unanswerable_wrong_test_df)

# Still need to reset index of DataFrame

data_nli_unanswerable_right_train_df = data_nli_unanswerable_right_train_df.reset_index(drop=True)
data_nli_unanswerable_right_val_df = data_nli_unanswerable_right_val_df.reset_index(drop=True)
data_nli_unanswerable_right_test_df = data_nli_unanswerable_right_test_df.reset_index(drop=True)

data_nli_unanswerable_wrong_train_df = data_nli_unanswerable_wrong_train_df.reset_index(drop=True)
data_nli_unanswerable_wrong_val_df = data_nli_unanswerable_wrong_val_df.reset_index(drop=True)
data_nli_unanswerable_wrong_test_df = data_nli_unanswerable_wrong_test_df.reset_index(drop=True)


# In[40]:


# For debug purpose

print("ENTAILMENT ANSWERABLE")
print("TRAIN:", len(data_nli_answerable_right_train_df))
print("VAL:", len(data_nli_answerable_right_val_df))
print("TEST:", len(data_nli_answerable_right_test_df))
print()

print("CONTRADICTION ANSWERABLE")
print("TRAIN:", len(data_nli_answerable_wrong_train_df))
print("VAL:", len(data_nli_answerable_wrong_val_df))
print("TEST:", len(data_nli_answerable_wrong_test_df))
print()

print("ENTAILMENT UN-ANSWERABLE")
print("TRAIN:", len(data_nli_unanswerable_right_train_df))
print("VAL:", len(data_nli_unanswerable_right_val_df))
print("TEST:", len(data_nli_unanswerable_right_test_df))
print()

print("CONTRADICTION UN-ANSWERABLE")
print("TRAIN:", len(data_nli_unanswerable_wrong_train_df))
print("VAL:", len(data_nli_unanswerable_wrong_val_df))
print("TEST:", len(data_nli_unanswerable_wrong_test_df))


# # Convert question-answer pair to hypothesis

# In[41]:


# Maybe we can try this approach

#nlp_tools_paraphraser = pipeline(task = TASK_PARAPHRASER_NAME, 
#                     model = MODEL_PARAPHRASER_NAME, 
#                     tokenizer = AutoTokenizer.from_pretrained(MODEL_PARAPHRASER_NAME, 
#                                                               model_max_length=512, 
#                                                               truncation=True))


# In[42]:


# This function useful for
# retrieve hypothesis from
# question and answer

def convert_question_and_answer_to_hypothesis(data, NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):
    
    data['hypothesis'] = ""
    hypothesis_array = list()
    
    for i in range(len(data)):
        
        if data['answer'][i] != "" and type(data['answer'][i]) == str:
            hypothesis_array.append(data['question'][i] + ' ' + data['answer'][i])
        else:
            hypothesis_array.append(data['question'][i] + ' ' + NO_ANSWER_STATEMENT)
        
        # Use this to decline no-answer-warning properties
        #hypothesis_array.append(data['question'][i] + ' ' + data['answer'][i])
        
        # Use this to use paraphraser
        #hypothesis_array.append(str(nlp_tools_paraphraser(data['question'][i] + ' ' + data['answer'][i])[0]['generated_text']))
    
    data['hypothesis'] = hypothesis_array
    
    return data


# In[43]:


data_nli_answerable_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_train_df)
data_nli_answerable_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_val_df)
data_nli_answerable_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_test_df)

data_nli_answerable_right_train_df = move_to_column_number(data_nli_answerable_right_train_df, "hypothesis", 3)
data_nli_answerable_right_val_df = move_to_column_number(data_nli_answerable_right_val_df, "hypothesis", 3)
data_nli_answerable_right_test_df = move_to_column_number(data_nli_answerable_right_test_df, "hypothesis", 3)


# In[44]:


data_nli_answerable_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_train_df)
data_nli_answerable_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_val_df)
data_nli_answerable_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_test_df)

data_nli_answerable_wrong_train_df = move_to_column_number(data_nli_answerable_wrong_train_df, "hypothesis", 3)
data_nli_answerable_wrong_val_df = move_to_column_number(data_nli_answerable_wrong_val_df, "hypothesis", 3)
data_nli_answerable_wrong_test_df = move_to_column_number(data_nli_answerable_wrong_test_df, "hypothesis", 3)


# In[45]:


data_nli_unanswerable_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_train_df)
data_nli_unanswerable_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_val_df)
data_nli_unanswerable_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_test_df)

data_nli_unanswerable_right_train_df = move_to_column_number(data_nli_unanswerable_right_train_df, "hypothesis", 3)
data_nli_unanswerable_right_val_df = move_to_column_number(data_nli_unanswerable_right_val_df, "hypothesis", 3)
data_nli_unanswerable_right_test_df = move_to_column_number(data_nli_unanswerable_right_test_df, "hypothesis", 3)


# In[46]:


data_nli_unanswerable_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_train_df)
data_nli_unanswerable_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_val_df)
data_nli_unanswerable_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_test_df)

data_nli_unanswerable_wrong_train_df = move_to_column_number(data_nli_unanswerable_wrong_train_df, "hypothesis", 3)
data_nli_unanswerable_wrong_val_df = move_to_column_number(data_nli_unanswerable_wrong_val_df, "hypothesis", 3)
data_nli_unanswerable_wrong_test_df = move_to_column_number(data_nli_unanswerable_wrong_test_df, "hypothesis", 3)


# # Assign the label: entailment & contradiction

# In[47]:


data_nli_answerable_right_train_df['label'] = 'entailment'
data_nli_answerable_right_val_df['label'] = 'entailment'
data_nli_answerable_right_test_df['label'] = 'entailment'

data_nli_answerable_right_train_df = move_to_column_number(data_nli_answerable_right_train_df, "label", 4)
data_nli_answerable_right_val_df = move_to_column_number(data_nli_answerable_right_val_df, "label", 4)
data_nli_answerable_right_test_df = move_to_column_number(data_nli_answerable_right_test_df, "label", 4)


# In[48]:


data_nli_answerable_wrong_train_df['label'] = 'contradiction'
data_nli_answerable_wrong_val_df['label'] = 'contradiction'
data_nli_answerable_wrong_test_df['label'] = 'contradiction'

data_nli_answerable_wrong_train_df = move_to_column_number(data_nli_answerable_wrong_train_df, "label", 4)
data_nli_answerable_wrong_val_df = move_to_column_number(data_nli_answerable_wrong_val_df, "label", 4)
data_nli_answerable_wrong_test_df = move_to_column_number(data_nli_answerable_wrong_test_df, "label", 4)


# In[49]:


data_nli_unanswerable_right_train_df['label'] = 'entailment'
data_nli_unanswerable_right_val_df['label'] = 'entailment'
data_nli_unanswerable_right_test_df['label'] = 'entailment'

data_nli_unanswerable_right_train_df = move_to_column_number(data_nli_unanswerable_right_train_df, "label", 4)
data_nli_unanswerable_right_val_df = move_to_column_number(data_nli_unanswerable_right_val_df, "label", 4)
data_nli_unanswerable_right_test_df = move_to_column_number(data_nli_unanswerable_right_test_df, "label", 4)


# In[50]:


data_nli_unanswerable_wrong_train_df['label'] = 'contradiction'
data_nli_unanswerable_wrong_val_df['label'] = 'contradiction'
data_nli_unanswerable_wrong_test_df['label'] = 'contradiction'

data_nli_unanswerable_wrong_train_df = move_to_column_number(data_nli_unanswerable_wrong_train_df, "label", 4)
data_nli_unanswerable_wrong_val_df = move_to_column_number(data_nli_unanswerable_wrong_val_df, "label", 4)
data_nli_unanswerable_wrong_test_df = move_to_column_number(data_nli_unanswerable_wrong_test_df, "label", 4)


# # Concat the right and wrong NLI to one NLI dataset

# In[51]:


data_nli_train_df_final = pd.concat([data_nli_answerable_right_train_df, 
                                     data_nli_answerable_wrong_train_df,
                                     data_nli_unanswerable_right_train_df,
                                     data_nli_unanswerable_wrong_train_df], axis=0, ignore_index=True)

data_nli_val_df_final = pd.concat([data_nli_answerable_right_val_df, 
                                   data_nli_answerable_wrong_val_df,
                                   data_nli_unanswerable_right_val_df,
                                   data_nli_unanswerable_wrong_val_df], axis=0, ignore_index=True)

data_nli_test_df_final = pd.concat([data_nli_answerable_right_test_df, 
                                    data_nli_answerable_wrong_test_df,
                                    data_nli_unanswerable_right_test_df,
                                    data_nli_unanswerable_wrong_test_df], axis=0, ignore_index=True)


# In[52]:


# For debug purpose,
# you can modify it too

def debug_data(data):
    
    for i in range(len(data)):
        
        print(f"Iteration: {i}")
        print(f"Answer: {data['answer'][i]}")
        
        print("NER")
        print(data['ner_tag_answer'][i])
        print(data['ner_tag_premise'][i])
        
        print("Chunking")
        print(data['chunking_tag_answer'][i])
        print(data['chunking_tag_premise'][i])
        print()

# debug_data(data_nli_train_df_final)
# debug_data(data_nli_val_df_final)
# debug_data(data_nli_test_df_final)


# In[53]:


# For debug purpose

print("TRAIN FINAL")
print(len(data_nli_train_df_final))
print()

print("VAL FINAL")
print(len(data_nli_val_df_final))
print()

print("TEST FINAL")
print(len(data_nli_test_df_final))


# # Convert to DataFrame format to CSV

# In[54]:


data_nli_train_df_final.to_csv(f"{NAME}_nli_train_df.csv", index=False)
data_nli_val_df_final.to_csv(f"{NAME}_nli_val_df.csv", index=False)
data_nli_test_df_final.to_csv(f"{NAME}_nli_test_df.csv", index=False)


# In[55]:


print("PROGRAM FINISHED")


# In[ ]:




