#!/usr/bin/env python
# coding: utf-8

# # Define tool and model of the tool

# In[2]:


#import os, torch
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
#os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:


import sys

TOOLS_NAME_NER = "ner"
MODEL_TOOLS_NAME_NER = "ageng-anugrah/indobert-large-p2-finetuned-ner"

TOOLS_NAME_POS = "token-classification"
MODEL_TOOLS_NAME_POS = "ageng-anugrah/indobert-large-p2-finetuned-chunking"

MODEL_SIMILARITY_NAME = "paraphrase-multilingual-mpnet-base-v2"

SAMPLE = sys.maxsize
# SAMPLE = 100


# # Import anything

# In[4]:


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

# In[5]:


print("PROGRAM STARTED")


# In[6]:

if __name__ == "__main__":


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

    # In[7]:


    seed_value = 42
    random.seed(seed_value)


    # In[8]:


    if SAMPLE == sys.maxsize:
        data_qas_train_df = pd.DataFrame(data_qas["train"][:SAMPLE])
        data_qas_val_df = pd.DataFrame(data_qas["validation"][:SAMPLE])
        data_qas_test_df = pd.DataFrame(data_qas["test"][:SAMPLE])

    else:
        data_qas_train_df = (pd.DataFrame(data_qas["train"])).sample(n=SAMPLE, random_state=42)
        data_qas_val_df = (pd.DataFrame(data_qas["validation"])).sample(n=SAMPLE, random_state=42)
        data_qas_test_df = (pd.DataFrame(data_qas["test"])).sample(n=SAMPLE, random_state=42)

        data_qas_train_df = data_qas_train_df.reset_index(drop=True)
        data_qas_val_df = data_qas_val_df.reset_index(drop=True)
        data_qas_test_df = data_qas_test_df.reset_index(drop=True)


    # ## Retrieve answer text only

    # In[9]:


    def retrieve_answer_text(data):
        for i in range(len(data)):
            data['answer'][i] = data['answer'][i]['text']
        return data


    # In[10]:


    data_qas_train_df = retrieve_answer_text(data_qas_train_df)
    data_qas_val_df = retrieve_answer_text(data_qas_val_df)
    data_qas_test_df = retrieve_answer_text(data_qas_test_df)


    # In[11]:


    x = data_qas_train_df[data_qas_train_df['answer'] == '']
    y = data_qas_val_df[data_qas_val_df['answer'] == '']
    z = data_qas_test_df[data_qas_test_df['answer'] == '']


    # In[12]:


    def returning_answer_form_and_answer_that_suitable(data):
        
        for i in range(len(data['question'])):
            
            question = data['question'][i]

            if "apa" in question.split():
                answer_form = "sentence"
                answer_ner = ""
            
            elif "siapa" in question.split():
                answer_form = "word"
                answer_ner = ""
            
            elif "kapan" in question.split():
                answer_form = "word"
                answer_ner = ""
            
            elif "dimana" in question.split():
                answer_form = "word"
                answer_ner = ""
            
            elif "mengapa" in question.split():
                answer_form = "sentence"
                answer_ner = ""
            
            elif "bagaimana" in question.split():
                answer_form = "sentence"
                answer_ner = ""
            
            elif "berapa" in question.split():
                answer_form = "word"
                answer_ner = ""
            
            else:
                answer_form = "word"
                answer_ner = ""


    # ## Delete all unanswerable row

    # In[13]:


    data_qas_train_df = data_qas_train_df[data_qas_train_df['answer'] != '']
    data_qas_val_df = data_qas_val_df[data_qas_val_df['answer'] != '']
    data_qas_test_df = data_qas_test_df[data_qas_test_df['answer'] != '']


    # ### Reset index number

    # In[14]:


    data_qas_train_df = data_qas_train_df.reset_index(drop=True)
    data_qas_val_df = data_qas_val_df.reset_index(drop=True)
    data_qas_test_df = data_qas_test_df.reset_index(drop=True)


    # ## Create NLI dataset from copy of QA dataset above

    # In[15]:


    #x = [
    #    {
    #     "Ir. Basuki Tjahaja Purnama, M.M. (EYD: Basuki Cahaya Purnama, nama Tionghoa: Zhōng Wànxué / 鍾萬學,[2] ), atau paling dikenal dengan panggilan Hakka Ahok (阿學), adalah Gubernur DKI Jakarta yang menjabat sejak 19 November 2014 hingga 9 Mei 2017.", 
    #     "Siapakah Gubernur DKI Jakarta yang menjabat sejak tahun 2014?", 
    #     "Basuki Tjahaja Purnama"
    #    }
    #]

    #x = pd.DataFrame(x, columns=["answer", "question", "context"])
    #x


    # In[16]:


    data_nli_train_df = data_qas_train_df.copy()
    data_nli_val_df = data_qas_val_df.copy()
    data_nli_test_df = data_qas_test_df.copy()


    # ## Convert context pair to premise (only renaming column)

    # In[17]:


    data_nli_train_df = data_nli_train_df.rename(columns={"context": "premise"})
    data_nli_val_df = data_nli_val_df.rename(columns={"context": "premise"})
    data_nli_test_df = data_nli_test_df.rename(columns={"context": "premise"})


    # # Add contradiction label cases

    # ## Import pipeline to create contradiction cases

    # In[18]:


    nlp_tools_ner = pipeline(task = TOOLS_NAME_NER, 
                        model = MODEL_TOOLS_NAME_NER, 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOOLS_NAME_NER, 
                                                                model_max_length=512, 
                                                                truncation=True),
                        aggregation_strategy = 'simple')


    # In[19]:


    nlp_tools_chunking = pipeline(task = TOOLS_NAME_POS, 
                        model = MODEL_TOOLS_NAME_POS, 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOOLS_NAME_POS, 
                                                                model_max_length=512, 
                                                                truncation=True),
                        aggregation_strategy = 'simple')


    # ## Add NER and chunking tag column in DataFrame

    # In[20]:


    def remove_space_after_number_and_punctuation(text):
        pattern = r'(\d+)\s*([.,])\s*(?=\S|$)'
        cleaned_text = re.sub(pattern, r'\1\2', text)
        return cleaned_text


    # In[21]:


    def add_row_tag(answer, tag, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

        if tag == "ner": tools=ner
        else: tools=chunking

        retrieved_from_tools = tools(answer)
        tag_answer_list = []
        
        if len(retrieved_from_tools) != 0:
            for i in retrieved_from_tools:
                tag_answer = (i['entity_group'], remove_space_after_number_and_punctuation(i['word']))
                tag_answer_list.append(tag_answer)
        else:
            tag_answer = ("NULL", answer)
            tag_answer_list.append(tag_answer)
            
        return tag_answer_list


    # In[22]:


    def add_premise_tag(data, tag, index, premise_array, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

        if tag == "ner": tools=ner
        else: tools=chunking
        
        if len(tools(data['premise'][index])) == 0:
            premise_array.append("NO TOKEN DETECTED")
        
        else:
            for j in tools(data['premise'][index]):
                tag_premise = (j['entity_group'], remove_space_after_number_and_punctuation(j['word']))
                premise_array.append(tag_premise)

        return premise_array


    # In[23]:


    def add_ner_and_chunking_all_tag(data):
        
        data['ner_tag_answer'] = ""
        data['chunking_tag_answer'] = ""
        
        data['ner_tag_premise'] = ""
        data['chunking_tag_premise'] = ""
        
        for i in tqdm(range(len(data))):
            
            answer = data['answer'][i]
            premise = data['premise'][i]
            
            ner_premise_array = []
            chunking_premise_array = []
                
            data['ner_tag_answer'][i] = add_row_tag(answer, "ner")
            data['chunking_tag_answer'][i] = add_row_tag(answer, "chunking")
                                                    
            data['ner_tag_premise'][i] = add_premise_tag(data, "ner", i, ner_premise_array)
            data['chunking_tag_premise'][i] = add_premise_tag(data, "chunking", i, chunking_premise_array)  
        
        return data


    # In[24]:


    data_nli_train_df = add_ner_and_chunking_all_tag(data_nli_train_df)
    data_nli_val_df = add_ner_and_chunking_all_tag(data_nli_val_df)
    data_nli_test_df = add_ner_and_chunking_all_tag(data_nli_test_df)


    # # Create wrong answer

    # This is the flow to create wrong answer:
    # 
    # 1. Check the NER and POS/Chunking labels of the right_answer and context/premise.
    # 
    # 2. Search and group NER and POS/Chunking labels that match the right_answer throughout the context/premise.
    # 
    # 3. Perform NER classification. There will be two branches here, namely:
    # 
    #    3a. If the NER of the right_answer can be detected, then calculate the distance using semantic similarity or word vectors between the right_answer and various possible wrong_answers with the same NER as the right_answer. Once done, proceed to the final wrong_answer.
    #    
    #    3b. If the NER of the right_answer cannot be detected (NULL) or context/premise does not contain any of NER of right_answer, then the POS/Chunking of the right_answer will be identified.
    #    
    # 4. Perform POS/Chunking classification. Continuation from point 3b. There will be two more branches:
    # 
    #    4a. If the POS/Chunking of the right_answer can be detected, then calculate the distance using semantic similarity or word vectors between the right_answer and various possible wrong_answers with the same POS/Chunking as the right_answer. Once done, proceed to the final wrong_answer.
    #    
    #    4b. If the POS/Chunking of the right_answer cannot be detected (NULL) or context/premise does not contain any of NER of right_answer, then the final wrong_answer will be chosen based on a random word (random_word) from the context/premise.

    # In[25]:


    model_similarity = SentenceTransformer(MODEL_SIMILARITY_NAME)

    def return_similarity_sorted_array(right_answer, sentence_array, model=model_similarity):
        
        embedding_right_answer = model.encode([right_answer], convert_to_tensor=True, device=device)
        embedding_sentence_array = model.encode(sentence_array, convert_to_tensor=True, device=device)
        
        cosine_scores = util.pytorch_cos_sim(embedding_right_answer, embedding_sentence_array)
        
        sorted_indices = cosine_scores.argsort(descending=True)[0]
        sorted_array = [sentence_array[i] for i in sorted_indices]
        
        return sorted_array


    # In[26]:


    def remove_values_with_hash(arr):
        return [item for item in arr if "#" not in item]


    # In[27]:


    def select_random_word(text):
        words = re.findall(r'\w+', text)
        random_word = random.choice(words)
        return random_word


    # In[28]:


    def grouping_same_tag(tag_answers, tag_premises, same_tag_array):

        for tag_premise in tag_premises:

            label_tag_premise = tag_premise[0]
            word_premise = tag_premise[1]

            for tag_answer in tag_answers:
                
                label_tag_answer = tag_answer[0]
                
                if label_tag_answer == label_tag_premise:
                    same_tag_array.append(word_premise)

        return remove_values_with_hash(same_tag_array)


    # In[29]:


    def remove_punctuation(text):
        return text.strip(string.punctuation)


    # In[30]:


    def filtering_plausible_answer(answer, plausible_answer_array):
        answer = answer.lower()
        
        plausible_answer_array = [item.lower().strip() for item in plausible_answer_array]
        plausible_answer_array = [string for string in plausible_answer_array if not contains_only_punctuation(string)]
        plausible_answer_array = [remove_punctuation(text) for text in plausible_answer_array]
        
        final_plausible_answer_array = []
        answer_words = set(remove_punctuation(text) for text in answer.split())
        
        for plausible_answer in plausible_answer_array:
            plausible_answer_words = set(plausible_answer.split())
            if not plausible_answer_words.intersection(answer_words):
                final_plausible_answer_array.append(plausible_answer)
        
        return final_plausible_answer_array


    # In[31]:


    def find_substring_span(long_string, substring):
        long_string = long_string.lower()
        substring = substring.lower()
        
        start_index = long_string.find(substring)
        
        if start_index != -1:
            end_index = start_index + len(substring) - 1
            return start_index, end_index
        else:
            return None

    def check_span_overlap(span1, span2):
        if span1 == None or span2 == None: return True # Exit plan
        else: return span1[0] <= span2[1] and span2[0] <= span1[1]

    def check_string_overlap(str1, str2):
        assert isinstance(str1, str)
        assert isinstance(str1, str)
        
        return (str1[-1] >= str2[0]) \
                or (str1 in str2) \
                or (str2 in str1)

    def contains_only_punctuation(text):
        return all(char in string.punctuation for char in text)


    # In[32]:


    def sorting_similarity(data, right_answer, index, tag, plausible_answer_array, premise):

        if tag == "ner": slice='same_ner_tag_answer'
        elif tag == "chunking": slice='same_chunking_tag_answer'
        else: slice=None

        # Find all the sorted (by similarity) plausible wrong answer, 
        # and remove hask & punctuation only answer
        if slice != None:
            wrong_answer_array = return_similarity_sorted_array(right_answer, data[slice][index])
        else:
            wrong_answer_array = return_similarity_sorted_array(right_answer, plausible_answer_array)
        
        plausible_answer_array = remove_values_with_hash(wrong_answer_array)
        plausible_answer_array = filtering_plausible_answer(right_answer, plausible_answer_array)
        
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
            wrong_answer = select_random_word(premise)
            
            if tag == "ner": 
                properties = "Detected (NER) wrong answer that is the SAME as the right answer, search random word from premise"
            elif tag == "chunking":
                properties = "Detected (Chunking) wrong answer that is the SAME as the right answer, search random word from premise"
            else:
                properties = "Detected (Random) wrong answer that is the SAME as the right answer, search random word from premise"
        
        assert isinstance(wrong_answer, str)
        assert isinstance(plausible_answer_array, list)
        
        return wrong_answer, plausible_answer_array, properties


    # In[33]:


    def create_wrong_answer(data):
        
        data['same_ner_tag_answer'] = ""
        data['same_chunking_tag_answer'] = ""
        data['wrong_answer'] = ""
        data['plausible_answer_based_on_method'] = ""
        data['properties'] = ""
        
        for i in tqdm(range(len(data))):
            
            right_answer = data['answer'][i]
            premise = data['premise'][i]

            same_ner_tag_answer_array = []
            same_chunking_tag_answer_array = []

            ner_tag_answer = data['ner_tag_answer'][i]
            ner_tag_premise = data['ner_tag_premise'][i]

            chunking_tag_answer = data['chunking_tag_answer'][i]
            chunking_tag_premise = data['chunking_tag_premise'][i]
            
            # Grouped with the same NER & Chunking group, between answer and word of premise
            data['same_ner_tag_answer'][i] = grouping_same_tag(ner_tag_answer,
                                                            ner_tag_premise,
                                                            same_ner_tag_answer_array)
            
            data['same_chunking_tag_answer'][i] = grouping_same_tag(chunking_tag_answer, 
                                                                    chunking_tag_premise, 
                                                                    same_chunking_tag_answer_array)
            
            # Start to create wrong answer
            plausible_answer_array = []

            # Perform NER classification
            # If the NER of the right_answer can be detected, then calculate the distance using semantic 
            # similarity or word vectors between the right_answer and various possible wrong_answers with 
            # the same NER as the right_answer. Once done, proceed to the final wrong_answer.
            if data['same_ner_tag_answer'][i] != []:
                wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                        i, "ner", plausible_answer_array, premise)
                
            # If the NER of the right_answer cannot be detected (NULL) or context/premise does not contain 
            # any of NER of right_answer, then the POS/Chunking of the right_answer will be identified.
            # Perform POS/Chunking classification
            else:
                # If the POS/Chunking of the right_answer can be detected, then calculate the distance 
                # using semantic similarity or word vectors between the right_answer and various possible 
                # wrong_answers with the same POS/Chunking as the right_answer. Once done, proceed to the 
                # final wrong_answer.
                if data['same_chunking_tag_answer'][i] != []:
                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "chunking", plausible_answer_array, premise)
                
                # If the POS/Chunking of the right_answer cannot be detected (NULL) or context/premise 
                # does not contain any of NER of right_answer, then the final wrong_answer will be chosen 
                # based on a random word (random_word) from the context/premise.
                else:
                    for chunking_tag in chunking_tag_premise:
                        plausible_answer_array.append(chunking_tag[1])

                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "none", plausible_answer_array, premise)
            data['properties'][i] = properties
            data['wrong_answer'][i] = wrong_answer
            data['plausible_answer_based_on_method'][i] = plausible_answer_array
                
        return data       


    # In[34]:


    def create_wrong_answer_with_removing_invalid_data(data):
        
        data['same_ner_tag_answer'] = ""
        data['same_chunking_tag_answer'] = ""
        data['wrong_answer'] = ""
        data['plausible_answer_based_on_method'] = ""
        data['properties'] = ""
        
        for i in tqdm(range(len(data))):
            
            right_answer = data['answer'][i]
            premise = data['premise'][i]

            same_ner_tag_answer_array = []
            same_chunking_tag_answer_array = []

            ner_tag_answer = data['ner_tag_answer'][i]
            ner_tag_premise = data['ner_tag_premise'][i]

            chunking_tag_answer = data['chunking_tag_answer'][i]
            chunking_tag_premise = data['chunking_tag_premise'][i]
            
            # Grouped with the same NER & Chunking group, between answer and word of premise
            data['same_ner_tag_answer'][i] = grouping_same_tag(ner_tag_answer,
                                                            ner_tag_premise,
                                                            same_ner_tag_answer_array)
            
            data['same_chunking_tag_answer'][i] = grouping_same_tag(chunking_tag_answer, 
                                                                    chunking_tag_premise, 
                                                                    same_chunking_tag_answer_array)
            
            # Start to create wrong answer
            plausible_answer_array = []
            
            # Golden rules: If same_NER isn't there, just drop it. If NER is NULL, check chunking
            
            if ner_tag_answer[0][0] == "NULL":
                if data['same_chunking_tag_answer'][i] != []:
                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "chunking", plausible_answer_array, premise)
                else:
                    for chunking_tag in chunking_tag_premise:
                        plausible_answer_array.append(chunking_tag[1])

                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "none", plausible_answer_array, premise)
                data['properties'][i] = properties
                data['wrong_answer'][i] = wrong_answer
                data['plausible_answer_based_on_method'][i] = plausible_answer_array
                continue

            # Perform NER classification
            # If the NER of the right_answer can be detected, then calculate the distance using semantic 
            # similarity or word vectors between the right_answer and various possible wrong_answers with 
            # the same NER as the right_answer. Once done, proceed to the final wrong_answer.
            if data['same_ner_tag_answer'][i] != [] and ner_tag_answer[0][0] != "NULL":
                wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                        i, "ner", plausible_answer_array, premise)
                
            # If the NER of the right_answer cannot be detected (NULL) or context/premise does not contain 
            # any of NER of right_answer, then drop that particular row data.
            else:
                data.drop(i, inplace=True)
                data.reset_index(drop=True)
                
            data['properties'][i] = properties
            data['wrong_answer'][i] = wrong_answer
            data['plausible_answer_based_on_method'][i] = plausible_answer_array
                
        return data       


    # In[35]:


    data_nli_train_df = create_wrong_answer_with_removing_invalid_data(data_nli_train_df)
    data_nli_val_df = create_wrong_answer_with_removing_invalid_data(data_nli_val_df)
    data_nli_test_df = create_wrong_answer_with_removing_invalid_data(data_nli_test_df)


    # In[36]:


    #for i in range(len(data_nli_train_df)):
        #print("Iterasi:", i)
        #print("Premise:", data_nli_train_df['premise'][i])
        #print("Right answer:", data_nli_train_df['answer'][i])
        #print("Wrong answer:", data_nli_train_df['wrong_answer'][i])
        #print("Same NER tag answer:", data_nli_train_df['same_ner_tag_answer'][i])
        #print("Same Chunking tag answer:", data_nli_train_df['same_chunking_tag_answer'][i])
        #print("Chunking tag premise", data_nli_train_df['chunking_tag_premise'][i])
        #print("Plausible answer:", data_nli_train_df['plausible_answer_based_on_method'][i])
        #print("Properties:", data_nli_train_df['properties'][i])
        #print("Overlap:", check_string_overlap(data_nli_train_df['answer'][i].lower(), data_nli_train_df['wrong_answer'][i].lower()))
        #print()


    # In[37]:


    #print(len(data_nli_train_df))
    #(len(data_nli_val_df))
    #print(len(data_nli_test_df))


    # In[38]:


    def test_create_wrong_answer(data):
        assert all(data['properties'] != '')
        assert all(data['wrong_answer'] != '')


    # In[39]:


    test_create_wrong_answer(data_nli_train_df)
    test_create_wrong_answer(data_nli_val_df)
    test_create_wrong_answer(data_nli_test_df)


    # # Split to two dataset: right dataset & wrong dataset

    # In[ ]:


    def move_to_column_number(data, column_name="hypothesis", column_num=3):

        cols = list(data.columns)
        cols.remove(column_name)
        cols.insert(column_num, column_name)

        data = data[cols]
        
        return data


    # In[ ]:


    columns_to_exclude = ['wrong_answer']

    data_nli_right_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
    data_nli_right_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
    data_nli_right_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()


    # In[ ]:


    columns_to_exclude = ['answer']

    data_nli_wrong_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
    data_nli_wrong_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
    data_nli_wrong_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()

    data_nli_wrong_train_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)
    data_nli_wrong_val_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)
    data_nli_wrong_test_df.rename(columns={'wrong_answer': 'answer'}, inplace=True)

    data_nli_wrong_train_df = move_to_column_number(data_nli_wrong_train_df, "answer", 2)
    data_nli_wrong_val_df = move_to_column_number(data_nli_wrong_val_df, "answer", 2)
    data_nli_wrong_test_df = move_to_column_number(data_nli_wrong_test_df, "answer", 2)


    # # Convert question-answer pair to hypothesis

    # In[ ]:


    def convert_question_and_answer_to_hypothesis(data):
        for i in range(len(data)):
            data['hypothesis'] = data['question'] + ' ' + data['answer']
        return data


    # In[ ]:


    data_nli_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_right_train_df)
    data_nli_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_right_val_df)
    data_nli_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_right_test_df)

    data_nli_right_train_df = move_to_column_number(data_nli_right_train_df, "hypothesis", 3)
    data_nli_right_val_df = move_to_column_number(data_nli_right_val_df, "hypothesis", 3)
    data_nli_right_test_df = move_to_column_number(data_nli_right_test_df, "hypothesis", 3)


    # In[ ]:


    data_nli_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_train_df)
    data_nli_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_val_df)
    data_nli_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_test_df)

    data_nli_wrong_train_df = move_to_column_number(data_nli_wrong_train_df, "hypothesis", 3)
    data_nli_wrong_val_df = move_to_column_number(data_nli_wrong_val_df, "hypothesis", 3)
    data_nli_wrong_test_df = move_to_column_number(data_nli_wrong_test_df, "hypothesis", 3)


    # # Add label: entailment & contradiction

    # In[ ]:


    data_nli_right_train_df['label'] = 'entailment'
    data_nli_right_val_df['label'] = 'entailment'
    data_nli_right_test_df['label'] = 'entailment'

    data_nli_right_train_df = move_to_column_number(data_nli_right_train_df, "label", 4)
    data_nli_right_train_df = move_to_column_number(data_nli_right_val_df, "label", 4)
    data_nli_right_train_df = move_to_column_number(data_nli_right_test_df, "label", 4)


    # In[ ]:


    data_nli_wrong_train_df['label'] = 'contradiction'
    data_nli_wrong_val_df['label'] = 'contradiction'
    data_nli_wrong_test_df['label'] = 'contradiction'

    data_nli_wrong_train_df = move_to_column_number(data_nli_wrong_train_df, "label", 4)
    data_nli_wrong_val_df = move_to_column_number(data_nli_wrong_val_df, "label", 4)
    data_nli_wrong_test_df = move_to_column_number(data_nli_wrong_test_df, "label", 4)


    # # Concat the right and wrong NLI to one NLI dataset

    # In[ ]:


    data_nli_train_df_final = pd.concat([data_nli_right_train_df, data_nli_wrong_train_df], axis=0, ignore_index=True)
    data_nli_val_df_final = pd.concat([data_nli_right_val_df, data_nli_wrong_val_df], axis=0, ignore_index=True)
    data_nli_test_df_final = pd.concat([data_nli_right_test_df, data_nli_wrong_test_df], axis=0, ignore_index=True)


    # In[ ]:


    # # Convert to DataFrame format to CSV

    # In[ ]:


    data_nli_train_df_final.to_csv("data_nli_train_df_drop.csv", index=False)
    data_nli_val_df_final.to_csv("data_nli_val_df_drop.csv", index=False)
    data_nli_test_df_final.to_csv("data_nli_test_df_drop.csv", index=False)


    # # Push to Hugging Face

    # In[ ]:


    """
    HUB_TOKEN = "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"
    USER = "muhammadravi251001"
    REPO = "idk-mrc-nli"

    api = HfApi()

    api.upload_file(
        path_or_fileobj="data_nli_train_df.csv",
        path_in_repo="data_nli_train_df.csv",
        repo_id=f"{USER}/{REPO}",
        token=HUB_TOKEN,
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="data_nli_val_df.csv",
        path_in_repo="data_nli_val_df.csv",
        repo_id=f"{USER}/{REPO}",
        token=HUB_TOKEN,
        repo_type="dataset",
    )

    api.upload_file(
        path_or_fileobj="data_nli_test_df.csv",
        path_in_repo="data_nli_test_df.csv",
        repo_id=f"{USER}/{REPO}",
        token=HUB_TOKEN,
        repo_type="dataset",
    )
    """


# In[ ]:


    print("PROGRAM FINISHED")

