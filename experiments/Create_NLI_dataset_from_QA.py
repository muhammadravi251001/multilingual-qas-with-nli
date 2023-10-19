#!/usr/bin/env python
# coding: utf-8

# # Define tool and model of the tool

# In[78]:


# get_ipython().system('nvidia-smi')


# Below, it is some settings to run in my local.

# In[79]:


import os, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# You can tweak your settings too in code below.

# In[80]:
import argparse
import sys

parser = argparse.ArgumentParser(description="Program untuk fine-tuning dataset QA")
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True)
parser.add_argument('-sa', '--sample', type=str, metavar='', required=True)
args = parser.parse_args()

DATA_NAME = args.data_name

if (args.sample) == "max":
    SAMPLE = sys.maxsize
else: 
    SAMPLE = int(args.sample)

NO_ANSWER_STATEMENT = "Tidak ada jawaban"

TASK_NER_NAME = "ner"
MODEL_NER_NAME = "ageng-anugrah/indobert-large-p2-finetuned-ner"

TASK_CHUNKING_NAME = "token-classification"
MODEL_CHUNKING_NAME = "ageng-anugrah/indobert-large-p2-finetuned-chunking"

MODEL_SIMILARITY_NAME = "paraphrase-multilingual-mpnet-base-v2"
URL_STOPWORD = "https://raw.githubusercontent.com/6/stopwords-json/master/stopwords-all.json"

TASK_PARAPHRASER_NAME = "text2text-generation"
MODEL_PARAPHRASER_NAME = ""

# # Import anything

# In[81]:


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

# In[82]:


print("PROGRAM STARTED")


# In[83]:


if (DATA_NAME == "squad-id"):
    conhelps = NusantaraConfigHelper()
    data_qas_id = conhelps.filtered(lambda x: 'squad_id' in x.dataset_name)[0].load_dataset()

    df_train = pd.DataFrame(data_qas_id['train'])
    df_test = pd.DataFrame(data_qas_id['validation'])

    cols = ['context', 'question', 'answer']
    new_df_test = pd.DataFrame(columns=cols)

    for i in tqdm(range(len(df_test['context']))):
        new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                        'question': df_test["question"][i], 
                                        'answer': {"text": eval(df_test["answer"][i][0])['text'], 
                                        "answer_start": eval(df_test["answer"][i][0])['answer_start'], 
                                        "answer_end": eval(df_test["answer"][i][0])['answer_end']}}, 
                                    ignore_index=True)

    cols = ['context', 'question', 'answer']
    new_df_train = pd.DataFrame(columns=cols)

    for i in tqdm(range(len(df_train['context']))):
        new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                        'question': df_train["question"][i], 
                                        'answer': {"text": eval(df_train["answer"][i][0])['text'], 
                                        "answer_start": eval(df_train["answer"][i][0])['answer_start'], 
                                        "answer_end": eval(df_train["answer"][i][0])['answer_end']}}, 
                                    ignore_index=True)

    train_final_df = new_df_train[:-11874]
    validation_final_df = new_df_train[-11874:]

    train_dataset = Dataset.from_dict(train_final_df)
    validation_dataset = Dataset.from_dict(validation_final_df)
    test_dataset = Dataset.from_dict(new_df_test)

    data_qas = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

elif (DATA_NAME == "idk-mrc"):
    conhelps = NusantaraConfigHelper()
    data_qas_id = conhelps.filtered(lambda x: 'idk_mrc' in x.dataset_name)[0].load_dataset()

    df_train = pd.DataFrame(data_qas_id['train'])
    df_validation = pd.DataFrame(data_qas_id['validation'])
    df_test = pd.DataFrame(data_qas_id['test'])

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

elif (DATA_NAME == "tydi-qa-id"):
    conhelps = NusantaraConfigHelper()
    data_qas_id = conhelps.filtered(lambda x: 'tydiqa_id' in x.dataset_name)[0].load_dataset()

    df_train = pd.DataFrame(data_qas_id['train'])
    df_validation = pd.DataFrame(data_qas_id['validation'])
    df_test = pd.DataFrame(data_qas_id['test'])

    cols = ['context', 'question', 'answer']
    new_df_train = pd.DataFrame(columns=cols)

    for i in range(len(df_train['context'])):
        answer_start = df_train['context'][i].index(df_train['label'][i])
        answer_end = answer_start + len(df_train['label'][i])
        new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                            'question': df_train["question"][i], 
                                            'answer': {"text": df_train["label"][i], 
                                                       "answer_start": answer_start, 
                                                       "answer_end": answer_end}}, 
                                                       ignore_index=True)

    cols = ['context', 'question', 'answer']
    new_df_val = pd.DataFrame(columns=cols)    

    for i in range(len(df_validation['context'])):
        answer_start = df_validation['context'][i].index(df_validation['label'][i])
        answer_end = answer_start + len(df_validation['label'][i])
        new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                        'question': df_validation["question"][i], 
                                        'answer': {"text": df_validation["label"][i], 
                                                   "answer_start": answer_start, 
                                                   "answer_end": answer_end}}, 
                                                   ignore_index=True)    

    cols = ['context', 'question', 'answer']
    new_df_test = pd.DataFrame(columns=cols)

    for i in range(len(df_test['context'])):
        answer_start = df_test['context'][i].index(df_test['label'][i])
        answer_end = answer_start + len(df_test['label'][i])
        new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                        'question': df_test["question"][i], 
                                        'answer': {"text": df_test["label"][i], 
                                                   "answer_start": answer_start, 
                                                   "answer_end": answer_end}}, 
                                                   ignore_index=True)

    train_dataset = Dataset.from_dict(new_df_train)
    validation_dataset = Dataset.from_dict(new_df_val)
    test_dataset = Dataset.from_dict(new_df_test)

    data_qas = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})


# # Convert to NLI, with hypothesis being just do concat question & answer

# ## Convert Dataset to DataFrame format

# In[84]:


# 42, the answer to life the universe and everything

seed_value = 42
random.seed(seed_value)


# In[85]:


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

# In[86]:


# Only retrieve answer text
# Because, we do not use answer_start
# and answer_end

def retrieve_answer_text(data):
    for i in range(len(data)):
        data['answer'][i] = data['answer'][i]['text']
    return data


# In[87]:


data_qas_train_df = retrieve_answer_text(data_qas_train_df)
data_qas_val_df = retrieve_answer_text(data_qas_val_df)
data_qas_test_df = retrieve_answer_text(data_qas_test_df)


# ## Create NLI dataset from copy of QA dataset above

# In[88]:


data_nli_train_df = data_qas_train_df.copy()
data_nli_val_df = data_qas_val_df.copy()
data_nli_test_df = data_qas_test_df.copy()


# In[89]:


data_qas_train_df


# In[90]:


data = {
    'context': ["Tanpa beasiswa, Ogilvy tidak bisa kuliah di Fettes atau Oxford University karena bisnis ayahnya terkena dampak depresi pertengahan dekade 1920-an. Namun, kuliahnya tidak berhasil dan ia meninggalkan Oxford untuk ke Paris pada tahun 1931 tempat ia menjadi chef magang di Majestic Hotel. Setelah setahun, ia kembali ke Skotlandia dan mulai menjual kompor masak AGA dari rumah ke rumah. Keberhasilannya dalam menjual kompor ini membuatnya dikenal sebagai karyawan, yang kemudian memintanya menulis manual instruksi, The Theory and Practice of Selling the AGA Cooker, untuk staf penjualan lainnya. Tiga puluh tahun kemudian, editor majalah Fortune menyebutnya sebagai manual instruksi penjualan terbaik yang pernah ditulis."],
    'question': ["Apa alasan Ogilvy tidak bisa kuliah di Fettes atau Oxford University?"],
    'answer': ["ogilvy tidak bisa kuliah di fettes atau oxford university karena bisnis ayahnya terkena dampak depresi pertengahan dekade 1920-an."]
}


# In[91]:


#data_debug = pd.DataFrame(data)
#data_debug


# ## Convert context pair to premise (only renaming column)

# In[92]:


# Renaming it, just for consistency

data_nli_train_df = data_nli_train_df.rename(columns={"context": "premise"})
data_nli_val_df = data_nli_val_df.rename(columns={"context": "premise"})
data_nli_test_df = data_nli_test_df.rename(columns={"context": "premise"})


# In[93]:


#data_debug = data_debug.rename(columns={"context": "premise"})


# # Add contradiction label cases

# ## Import pipeline to create contradiction cases

# In[94]:


nlp_tools_ner = pipeline(task = TASK_NER_NAME, 
                     model = MODEL_NER_NAME, 
                     tokenizer = AutoTokenizer.from_pretrained(MODEL_NER_NAME, 
                                                               model_max_length=512, 
                                                               truncation=True),
                     aggregation_strategy = 'simple')


# In[95]:


nlp_tools_chunking = pipeline(task = TASK_CHUNKING_NAME, 
                     model = MODEL_CHUNKING_NAME, 
                     tokenizer = AutoTokenizer.from_pretrained(MODEL_CHUNKING_NAME, 
                                                               model_max_length=512, 
                                                               truncation=True),
                     aggregation_strategy = 'simple')


# ## Add NER and chunking tag column in DataFrame

# In[96]:


# This code useful for cleaning the data (text)

def remove_space_after_number_and_punctuation(text):
    pattern = r'(\d+)\s*([.,])\s*(?=\S|$)'
    cleaned_text = re.sub(pattern, r'\1\2', text)
    return cleaned_text


# In[97]:


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


# In[98]:


# Function for clean the text off punctuation

def remove_punctuation(text):
    cleaned_text = text.lstrip(string.punctuation)
    cleaned_text = cleaned_text.rstrip(string.punctuation)
    return cleaned_text


# In[99]:


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
                
                lower_answer = answer.lower()
                lower_word_from_premise_tag = word_from_premise_tag.lower()

                # With assumption, that I do not dividing label when
                # there is more than one label in one word answer.
                # Instead, I give a NULL.

                if lower_word_from_premise_tag == lower_answer:
                    tag_answer = (label_from_premise_tag, word_from_premise_tag)
                    break

                # Or, I could do this: to reducing NULL label 
                # with subset of string not really with the entire string.
                
                elif lower_answer in lower_word_from_premise_tag or lower_word_from_premise_tag in lower_answer:
                    tag_answer = (label_from_premise_tag, answer)
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
                
                lower_answer = answer.lower()
                lower_word_from_premise_tag = word_from_premise_tag.lower()
                
                if lower_answer in lower_word_from_premise_tag:
                    tag_answer = (label_from_premise_tag, answer)
                    tag_answer_list.append(tag_answer)
                    break
            
            # Use for and then direct else (for-else),
            # if for-loop above not getting the break statement
            
            else:
                tag_answer = ("NULL", answer)
                tag_answer_list.append(tag_answer)
        
    return tag_answer_list


# In[100]:


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


# In[101]:


data_nli_train_df = add_ner_and_chunking_all_tag(data_nli_train_df)
data_nli_val_df = add_ner_and_chunking_all_tag(data_nli_val_df)
data_nli_test_df = add_ner_and_chunking_all_tag(data_nli_test_df)


# In[102]:


#data_debug = add_ner_and_chunking_all_tag(data_debug)
#data_debug


# # Create wrong answer

# In[103]:


# This function useful for sorting the closest distance
# by using embedding

model_similarity = SentenceTransformer(MODEL_SIMILARITY_NAME)

def return_similarity_sorted_array(right_answer, sentence_array, model=model_similarity, batch_size=32, device=device):
    sorted_array = []
    
    for i in range(0, len(sentence_array), batch_size):
        batch = sentence_array[i:i + batch_size]
        
        embedding_right_answer = model.encode([right_answer] * len(batch), convert_to_tensor=True, device=device)
        embedding_sentence_array = model.encode(batch, convert_to_tensor=True, device=device)

        # Using cosine scores to calculate
        cosine_scores = util.pytorch_cos_sim(embedding_right_answer, embedding_sentence_array)

        sorted_indices = cosine_scores.argsort(descending=True)[0]
        sorted_batch = [batch[i] for i in sorted_indices]
        sorted_array.extend(sorted_batch)

    return sorted_array



# In[104]:


# This function useful for
# removing value with hash.
# Because, from label-tagging before
# Some data have a hash symbol, because
# that data was part of a word fragment

def remove_values_with_hash(arr):
    return [item for item in arr if "#" not in item]


# In[105]:


# Retrieve stopword from all language

response = requests.get(URL_STOPWORD)

if response.status_code == 200:
    stopword_data = response.json()
else:
    print("Failed to download stopword JSON.")

stopword_data = set([item for sublist in list(stopword_data.values()) for item in sublist])


# In[106]:


# This function just retrieve random word
# of entire premise

def select_random_word(text, answer, stopword_data=stopword_data):

    words = re.findall(r'\w+', text)
    
    # Filtering to remove stopword and punctuation
    filtered_words = [word for word in words if word not in stopword_data and word not in string.punctuation]
    
    # If filtered words less than answer
    # only take one word as random word

    splitted_answer = answer.split()
    
    if len(filtered_words) < len(splitted_answer):
        random_word = random.choice(filtered_words)
    
    # But, if filtered words NOT less than answer
    # take a same length word as a random word
    # with the same order as filtered words
    
    else:
        
        start_index = random.randint(0, len(filtered_words) - len(splitted_answer))
        random_word_array = filtered_words[start_index : start_index + len(splitted_answer)]
        random_word = ' '.join(random_word_array)
    
    return random_word.strip()


# In[107]:


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
            
            matching_words = []
            
            j = 0
            
            while i + j < len(premise) and j < len(answer_labels) and premise[i + j][0] == answer_labels[j]:
                matching_words.append(premise[i + j][1])
                j += 1
            
            if j == len(answer_labels):
                results.append(" ".join(matching_words))
            
        i += 1
    
    return results


# In[108]:


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

    # Still, filter value with hash
    
    return remove_values_with_hash(same_tag_array)


# In[109]:


# This function useful for
# checking text if only
# contain punctuation, no words at all 

def contains_only_punctuation(text):
    return all(char in string.punctuation for char in text)


# In[110]:


# This function useful for
# filter overlapping right answer and wrong answer
# that provided in plausible answer

def filtering_plausible_answer(answer, plausible_answer_array):
    
    if type(plausible_answer_array) == str: 
        plausible_answer_array = list([plausible_answer_array])
    
    plausible_answer_array = [remove_punctuation(item.strip()) \
                            for item in plausible_answer_array \
                                if not contains_only_punctuation(item.strip())]
    
    final_plausible_answer_array = list()

    splitted_answer = answer.split()
    answer_words = set(remove_punctuation(text) for text in splitted_answer)
    
    # For check overlapping answer, using set of word,
    # and so, check for intersection
    
    for plausible_answer in plausible_answer_array:

        spliited_plausible_answer = plausible_answer.split()
        plausible_answer_words = set(spliited_plausible_answer)
        
        if not plausible_answer_words.intersection(answer_words):
            if not all(word in answer for word in spliited_plausible_answer):
                final_plausible_answer_array.append(plausible_answer)
    
    return final_plausible_answer_array


# In[111]:


# This function useful for
# detecting number, date, time
# to give plausible answer more
# "make sense" answer

def is_number(input_str):
    pattern = r'\d'
    return bool(re.search(pattern, input_str))

def is_date(input_str):
    pattern = r'\b\d{1,2}[/\s](\d{1,2}|\w+)[/\s]\d{4}\b'
    return bool(re.search(pattern, input_str))

def is_time(input_str):
    pattern = r'\b\d{1,2}[:.]\d{2}(:\d{2})?\b'
    return bool(re.search(pattern, input_str))

def check_regex(right_answer, plausible_answer_array):
    
    if is_date(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_date(item)]
    
    elif is_time(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_time(item)]
        
    elif is_number(right_answer):
        plausible_answer_array = [item for item in plausible_answer_array if is_number(item)]
        
    else:
        plausible_answer_array = [item for item in plausible_answer_array if (not is_number(item) or 
                                                                              not is_date(item) or 
                                                                              not is_time(item)
                                                                             )]
    
    return plausible_answer_array


# In[112]:


# This function useful for
# overlap checking
# after select random word

def overlap_checking_with_random_word(premise, 
                                      right_answer, 
                                      max_iter=10, 
                                      word_threshold=3, 
                                      NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):
    
    # Selecting wrong answer from random word in premise
    wrong_answer = select_random_word(premise, right_answer)

    # If that random word is overlapping to right answer,
    # iterate again until it is not overlap again until reach the max counter

    counter = 0

    while True:

        counter += 1
        
        # Restricting to be extractive model (substring must matched!)
        
        if len(filtering_plausible_answer(right_answer, wrong_answer)) > 0 \
            and wrong_answer in premise:
            break

        # If it still detect overlapped right answer,
        # just assign it with NO_ANSWER_STATEMENT.

        if counter == max_iter or len(wrong_answer) < word_threshold:
            wrong_answer = NO_ANSWER_STATEMENT
            break

        wrong_answer = select_random_word(premise, right_answer)
    
    return wrong_answer


# In[118]:


# This function useful for
# sorting similarity and
# to give final wrong answer 
# and properties

def return_wrong_and_plausible(data, right_answer, index, tag, plausible_answer_array, premise,
                       NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):

    if tag == "ner": slice = 'same_ner_tag_answer'
    elif tag == "chunking": slice = 'same_chunking_tag_answer'
    
    # Find all the sorted (by similarity) plausible wrong answer, 
    # and remove hask & punctuation only answer
    
    wrong_answer_array = return_similarity_sorted_array(right_answer, data[slice][index])
    
    # Below, do the filtering to plausible answer

    plausible_answer_array = remove_values_with_hash(wrong_answer_array)
    plausible_answer_array = filtering_plausible_answer(right_answer, plausible_answer_array)
    plausible_answer_array = check_regex(right_answer, plausible_answer_array)

    if len(plausible_answer_array) > 0:
        
        # Only return the most similar to right_answer
        wrong_answer = plausible_answer_array[0].strip()
        
        if tag == "ner": 
            properties = "IDENTICAL NER labels were found, and the highest similarity score same NER array was selected"
        
        elif tag == "chunking":
            properties = "IDENTICAL Chunking labels were found, and the highest similarity score from same Chunking array was selected"
    
    else:
        
        # Return wrong answer from random word in premise with overlap checking
        wrong_answer = overlap_checking_with_random_word(premise, right_answer)
        
        if tag == "ner": 
            properties = "Detected (NER) wrong answer that is the SAME as the right answer, search random word from premise"
        
        elif tag == "chunking":
            properties = "Detected (Chunking) wrong answer that is the SAME as the right answer, search random word from premise"
    
    # Still need to check/assert the wrong answer
    # and the plausible answer type
    
    assert isinstance(wrong_answer, str)
    assert isinstance(plausible_answer_array, list)
    
    return wrong_answer, plausible_answer_array, properties


# In[119]:


# This function useful for
# matching regex in the first
# section on create wrong answer
# to detect: number, date, and time

def matching_regex(right_answer, chunking_tag_premise):
    
    plausible_answer_array = []

    for _, word in chunking_tag_premise:
        
        word = remove_punctuation(word)
        right_answer = remove_punctuation(right_answer)
        
        if (word != right_answer) and (word not in right_answer):

            if is_date(word) and is_date(right_answer):
                plausible_answer_array.append(word)

            if is_time(word) and is_time(right_answer):
                plausible_answer_array.append(word)
                
            if is_number(word) and is_number(right_answer):
                plausible_answer_array.append(word)

    return plausible_answer_array


# In[120]:


# This function useful for
# cleaning the reference off the premise

def cleaning_premise(premise):
    cleaned_premise = re.sub(r'\[.*?\]', '', premise)
    cleaned_premise = re.sub(r'jmpl\|200px\|', '', cleaned_premise)
    return cleaned_premise


# In[121]:


# This function is the main idea to create wrong answer
# Though, this function is helper function

def create_wrong_answer(data, NO_ANSWER_STATEMENT=NO_ANSWER_STATEMENT):
    
    data['same_ner_tag_answer'] = ""
    data['same_chunking_tag_answer'] = ""
    data['wrong_answer'] = ""
    data['no_answer'] = ""
    data['plausible_answer_based_on_method'] = ""
    data['properties'] = ""
    
    #print("FOR A DEBUG PUPROSE")
    
    for i in tqdm(range(len(data))):
        
        #print(f"Iteration: {i}")
        #print(f"Premise: {data['premise'][i]}")
        #print(f"Question: {data['question'][i]}")
        #print(f"Right answer: {data['answer'][i]}")
        
        right_answer = data['answer'][i]
        premise = cleaning_premise(data['premise'][i])

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
            
            plausible_answer_array = matching_regex(right_answer, chunking_tag_premise)
            plausible_answer_array = filtering_plausible_answer(right_answer, plausible_answer_array)
            
            if len(plausible_answer_array) > 0:
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
            # selected random word from premise.
            
            else:
                properties = "No same tag detected, search random word from premise"
                wrong_answer = overlap_checking_with_random_word(premise, right_answer)
                plausible_answer_array = list()
        
        data['properties'][i] = properties
        data['wrong_answer'][i] = wrong_answer
        data['no_answer'][i] = NO_ANSWER_STATEMENT
        data['plausible_answer_based_on_method'][i] = list(set(plausible_answer_array))
            
    return data       


# In[124]:


data_nli_train_df = create_wrong_answer(data_nli_train_df)
data_nli_val_df = create_wrong_answer(data_nli_val_df)
data_nli_test_df = create_wrong_answer(data_nli_test_df)


# In[123]:


#data_debug = create_wrong_answer(data_debug)
#data_debug


# In[357]:


#print("Right answer:", data_debug['answer'][0])
#print()
#print(data_debug['plausible_answer_based_on_method'][0])
#print("Wrong answer:", data_debug['wrong_answer'][0])


# In[297]:


#print("Premise:", data_debug['premise'][0])
#print("Question:", data_debug['question'][0])
#print()
#print(data_debug['ner_tag_premise'][0])
#print(data_debug['chunking_tag_premise'][0])


# # Split to two dataset: right dataset & wrong dataset

# In[ ]:


# This method is just only
# for aesthetics of column number

def move_to_column_number(data, column_name="hypothesis", column_num=3):

    cols = list(data.columns)
    cols.remove(column_name)
    cols.insert(column_num, column_name)

    data = data[cols]
    
    return data


# In[ ]:


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


# In[ ]:


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


# In[ ]:


# Creating unanswerable right (entailment label) and no-answer dataset

columns_to_exclude = ['wrong_answer', 'no_answer']

data_nli_unanswerable_right_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_right_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
data_nli_unanswerable_right_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()

data_nli_unanswerable_right_train_df = data_nli_unanswerable_right_train_df[data_nli_unanswerable_right_train_df['answer'] == '']
data_nli_unanswerable_right_val_df = data_nli_unanswerable_right_val_df[data_nli_unanswerable_right_val_df['answer'] == '']
data_nli_unanswerable_right_test_df = data_nli_unanswerable_right_test_df[data_nli_unanswerable_right_test_df['answer'] == '']


# In[ ]:


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


# In[ ]:


# Rather than duplicating the no-answer statement, 
# it's better to remove the excessing row ones.

def balancing_data(data1, data2):
    
    if len(data1) > len(data2):
        data1 = data1.sample(n=len(data2))
    
    elif len(data1) < len(data2):
        data2 = data2.sample(n=len(data1))
        
    return data1, data2


# In[ ]:


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


# In[ ]:


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

# In[ ]:


# Maybe we can try this approach

#nlp_tools_paraphraser = pipeline(task = TASK_PARAPHRASER_NAME, 
#                     model = MODEL_PARAPHRASER_NAME, 
#                     tokenizer = AutoTokenizer.from_pretrained(MODEL_PARAPHRASER_NAME, 
#                                                               model_max_length=512, 
#                                                               truncation=True))


# In[ ]:


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


# In[ ]:


data_nli_answerable_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_train_df)
data_nli_answerable_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_val_df)
data_nli_answerable_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_right_test_df)

data_nli_answerable_right_train_df = move_to_column_number(data_nli_answerable_right_train_df, "hypothesis", 3)
data_nli_answerable_right_val_df = move_to_column_number(data_nli_answerable_right_val_df, "hypothesis", 3)
data_nli_answerable_right_test_df = move_to_column_number(data_nli_answerable_right_test_df, "hypothesis", 3)


# In[ ]:


data_nli_answerable_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_train_df)
data_nli_answerable_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_val_df)
data_nli_answerable_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_answerable_wrong_test_df)

data_nli_answerable_wrong_train_df = move_to_column_number(data_nli_answerable_wrong_train_df, "hypothesis", 3)
data_nli_answerable_wrong_val_df = move_to_column_number(data_nli_answerable_wrong_val_df, "hypothesis", 3)
data_nli_answerable_wrong_test_df = move_to_column_number(data_nli_answerable_wrong_test_df, "hypothesis", 3)


# In[ ]:


data_nli_unanswerable_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_train_df)
data_nli_unanswerable_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_val_df)
data_nli_unanswerable_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_right_test_df)

data_nli_unanswerable_right_train_df = move_to_column_number(data_nli_unanswerable_right_train_df, "hypothesis", 3)
data_nli_unanswerable_right_val_df = move_to_column_number(data_nli_unanswerable_right_val_df, "hypothesis", 3)
data_nli_unanswerable_right_test_df = move_to_column_number(data_nli_unanswerable_right_test_df, "hypothesis", 3)


# In[ ]:


data_nli_unanswerable_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_train_df)
data_nli_unanswerable_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_val_df)
data_nli_unanswerable_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_unanswerable_wrong_test_df)

data_nli_unanswerable_wrong_train_df = move_to_column_number(data_nli_unanswerable_wrong_train_df, "hypothesis", 3)
data_nli_unanswerable_wrong_val_df = move_to_column_number(data_nli_unanswerable_wrong_val_df, "hypothesis", 3)
data_nli_unanswerable_wrong_test_df = move_to_column_number(data_nli_unanswerable_wrong_test_df, "hypothesis", 3)


# # Assign the label: entailment & contradiction

# In[ ]:


data_nli_answerable_right_train_df['label'] = 'entailment'
data_nli_answerable_right_val_df['label'] = 'entailment'
data_nli_answerable_right_test_df['label'] = 'entailment'

data_nli_answerable_right_train_df = move_to_column_number(data_nli_answerable_right_train_df, "label", 4)
data_nli_answerable_right_val_df = move_to_column_number(data_nli_answerable_right_val_df, "label", 4)
data_nli_answerable_right_test_df = move_to_column_number(data_nli_answerable_right_test_df, "label", 4)


# In[ ]:


data_nli_answerable_wrong_train_df['label'] = 'contradiction'
data_nli_answerable_wrong_val_df['label'] = 'contradiction'
data_nli_answerable_wrong_test_df['label'] = 'contradiction'

data_nli_answerable_wrong_train_df = move_to_column_number(data_nli_answerable_wrong_train_df, "label", 4)
data_nli_answerable_wrong_val_df = move_to_column_number(data_nli_answerable_wrong_val_df, "label", 4)
data_nli_answerable_wrong_test_df = move_to_column_number(data_nli_answerable_wrong_test_df, "label", 4)


# In[ ]:


data_nli_unanswerable_right_train_df['label'] = 'entailment'
data_nli_unanswerable_right_val_df['label'] = 'entailment'
data_nli_unanswerable_right_test_df['label'] = 'entailment'

data_nli_unanswerable_right_train_df = move_to_column_number(data_nli_unanswerable_right_train_df, "label", 4)
data_nli_unanswerable_right_val_df = move_to_column_number(data_nli_unanswerable_right_val_df, "label", 4)
data_nli_unanswerable_right_test_df = move_to_column_number(data_nli_unanswerable_right_test_df, "label", 4)


# In[ ]:


data_nli_unanswerable_wrong_train_df['label'] = 'contradiction'
data_nli_unanswerable_wrong_val_df['label'] = 'contradiction'
data_nli_unanswerable_wrong_test_df['label'] = 'contradiction'

data_nli_unanswerable_wrong_train_df = move_to_column_number(data_nli_unanswerable_wrong_train_df, "label", 4)
data_nli_unanswerable_wrong_val_df = move_to_column_number(data_nli_unanswerable_wrong_val_df, "label", 4)
data_nli_unanswerable_wrong_test_df = move_to_column_number(data_nli_unanswerable_wrong_test_df, "label", 4)


# # Concat the right and wrong NLI to one NLI dataset

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

# In[ ]:


data_nli_train_df_final.to_csv(f"{DATA_NAME}_nli_train_df.csv", index=False)
data_nli_val_df_final.to_csv(f"{DATA_NAME}_nli_val_df.csv", index=False)
data_nli_test_df_final.to_csv(f"{DATA_NAME}_nli_test_df.csv", index=False)


# In[ ]:


print("PROGRAM FINISHED")


# In[ ]:




