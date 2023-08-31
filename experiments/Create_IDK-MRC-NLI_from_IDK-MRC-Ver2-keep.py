#!/usr/bin/env python
# coding: utf-8

# # Define tool and model of the tool

# In[1]:


import sys

TOOLS_NAME_NER = "ner"
MODEL_TOOLS_NAME_NER = "ageng-anugrah/indobert-large-p2-finetuned-ner"

TOOLS_NAME_POS = "token-classification"
MODEL_TOOLS_NAME_POS = "ageng-anugrah/indobert-large-p2-finetuned-chunking"

MODEL_SIMILARITY_NAME = "paraphrase-multilingual-mpnet-base-v2"

# SAMPLE = sys.maxsize
SAMPLE = 50


# # Import anything

# In[2]:


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

# In[3]:
if __name__ == "__main__":

    print("PROGRAM STARTED")

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

    # In[4]:


    seed_value = 42
    random.seed(seed_value)


    # In[5]:


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

    # In[6]:


    def retrieve_answer_text(data):
        for i in range(len(data)):
            data['answer'][i] = data['answer'][i]['text']
        return data


    # In[7]:


    data_qas_train_df = retrieve_answer_text(data_qas_train_df)
    data_qas_val_df = retrieve_answer_text(data_qas_val_df)
    data_qas_test_df = retrieve_answer_text(data_qas_test_df)


    # In[8]:


    x = data_qas_train_df[data_qas_train_df['answer'] == '']
    y = data_qas_val_df[data_qas_val_df['answer'] == '']
    z = data_qas_test_df[data_qas_test_df['answer'] == '']


    # In[9]:


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

    # In[10]:


    data_qas_train_df = data_qas_train_df[data_qas_train_df['answer'] != '']
    data_qas_val_df = data_qas_val_df[data_qas_val_df['answer'] != '']
    data_qas_test_df = data_qas_test_df[data_qas_test_df['answer'] != '']


    # ### Reset index number

    # In[11]:


    data_qas_train_df = data_qas_train_df.reset_index(drop=True)
    data_qas_val_df = data_qas_val_df.reset_index(drop=True)
    data_qas_test_df = data_qas_test_df.reset_index(drop=True)


    # ## Create NLI dataset from copy of QA dataset above

    # In[12]:


    #x = [
    #    {
    #     "Ir. Basuki Tjahaja Purnama, M.M. (EYD: Basuki Cahaya Purnama, nama Tionghoa: Zhōng Wànxué / 鍾萬學,[2] ), atau paling dikenal dengan panggilan Hakka Ahok (阿學), adalah Gubernur DKI Jakarta yang menjabat sejak 19 November 2014 hingga 9 Mei 2017.", 
    #     "Siapakah Gubernur DKI Jakarta yang menjabat sejak tahun 2014?", 
    #     "Basuki Tjahaja Purnama"
    #    }
    #]

    #x = pd.DataFrame(x, columns=["answer", "question", "context"])
    #x


    # In[13]:


    data_nli_train_df = data_qas_train_df.copy()
    data_nli_val_df = data_qas_val_df.copy()
    data_nli_test_df = data_qas_test_df.copy()


    # ## Convert context pair to premise (only renaming column)

    # In[14]:


    data_nli_train_df = data_nli_train_df.rename(columns={"context": "premise"})
    data_nli_val_df = data_nli_val_df.rename(columns={"context": "premise"})
    data_nli_test_df = data_nli_test_df.rename(columns={"context": "premise"})


    # # Add contradiction label cases

    # ## Import pipeline to create contradiction cases

    # In[15]:


    nlp_tools_ner = pipeline(task = TOOLS_NAME_NER, 
                        model = MODEL_TOOLS_NAME_NER, 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOOLS_NAME_NER, 
                                                                model_max_length=512, 
                                                                truncation=True),
                        aggregation_strategy = 'simple')


    # In[16]:


    nlp_tools_chunking = pipeline(task = TOOLS_NAME_POS, 
                        model = MODEL_TOOLS_NAME_POS, 
                        tokenizer = AutoTokenizer.from_pretrained(MODEL_TOOLS_NAME_POS, 
                                                                model_max_length=512, 
                                                                truncation=True),
                        aggregation_strategy = 'simple')


    # ## Add NER and chunking tag column in DataFrame

    # In[17]:


    def add_row_tag(answer, tag, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

        if tag == "ner": tools=ner
        else: tools=chunking

        retrieved_from_tools = tools(answer)
        tag_answer_list = []
        
        if len(retrieved_from_tools) != 0:
            for i in retrieved_from_tools:
                tag_answer = (i['entity_group'], i['word'])
                tag_answer_list.append(tag_answer)
        else:
            tag_answer = ("NULL", answer)
            tag_answer_list.append(tag_answer)
            
        return tag_answer_list


    # In[18]:


    def add_premise_tag(data, tag, index, premise_array, ner=nlp_tools_ner, chunking=nlp_tools_chunking):

        if tag == "ner": tools=ner
        else: tools=chunking
        
        if len(tools(data['premise'][index])) == 0:
            premise_array.append("NO TOKEN DETECTED")
        
        else:
            for j in tools(data['premise'][index]):
                tag_premise = (j['entity_group'], j['word'])
                premise_array.append(tag_premise)

        return premise_array


    # In[19]:


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


    # In[20]:


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

    # In[21]:


    model_similarity = SentenceTransformer(MODEL_SIMILARITY_NAME)

    def return_similarity_sorted_array(right_answer, sentence_array, model=model_similarity):
        
        embedding_right_answer = model.encode([right_answer], convert_to_tensor=True)
        embedding_sentence_array = model.encode(sentence_array, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(embedding_right_answer, embedding_sentence_array)
        
        sorted_indices = cosine_scores.argsort(descending=True)[0]
        sorted_array = [sentence_array[i] for i in sorted_indices]
        
        return sorted_array


    # In[22]:


    def remove_values_with_hash(arr):
        return [item for item in arr if "#" not in item]


    # In[23]:


    def select_random_word(text):
        words = re.findall(r'\w+', text)
        random_word = random.choice(words)
        return random_word


    # In[24]:


    def grouping_same_tag(tag_answers, tag_premises, same_tag_array):

        for tag_premise in tag_premises:

            label_tag_premise = tag_premise[0]
            word_premise = tag_premise[1]

            for tag_answer in tag_answers:
                
                label_tag_answer = tag_answer[0]
                
                if label_tag_answer == label_tag_premise:
                    same_tag_array.append(word_premise)

        return remove_values_with_hash(same_tag_array)


    # In[25]:


    def sorting_similarity(data, right_answer, index, tag, plausible_answer_array):

        if tag == "ner": slice='same_ner_tag_answer'
        elif tag == "chunking": slice='same_chunking_tag_answer'
        else: slice=None
            
        plausible_answer_array = [item for item in plausible_answer_array \
                                if item not in [right_answer.lower()]]

        # Find all the sorted (by similarity) plausible wrong answer, 
        # and remove hask & punctuation only answer
        if slice != None:
            wrong_answer_array = return_similarity_sorted_array(right_answer, data[slice][index])
        else:
            wrong_answer_array = return_similarity_sorted_array(right_answer, plausible_answer_array)
        
        plausible_answer_array = remove_values_with_hash(wrong_answer_array)
        plausible_answer_array = [string for string in plausible_answer_array \
                                        if not contains_only_punctuation(string)]

        # Only return the most similar to right_answer
        wrong_answer = plausible_answer_array[0]
        
        assert isinstance(wrong_answer, str)
        assert isinstance(plausible_answer_array, list)
        
        if tag == "ner": 
            properties = """IDENTICAL NER labels were found, and the highest similarity \
                                        score same NER array was selected"""
        elif tag == "chunking":
            properties = """IDENTICAL Chunking labels were found, and the highest similarity \
                                            score from same Chunking array was selected"""
        else:
            properties = """NO CHUNKING labels were found, and the highest similarity score \
                                            from plausible answer was selected"""
        
        return wrong_answer, plausible_answer_array, properties


    # In[26]:


    def find_substring_span(long_string, substring):
        long_string = long_string.lower()
        substring = substring.lower()
        
        start_index = long_string.find(substring)
        
        if start_index != -1:
            end_index = start_index + len(substring) - 1
            return start_index, end_index
        else:
            return None


    # In[27]:


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


    # In[28]:


    def replace_same_answer(right_answer, 
                            wrong_answer, 
                            premise, 
                            plausible_answer_array):
        
        # Removing right answer & wrong answer in this particular time
        plausible_answer_array = [item for item in plausible_answer_array \
                                if item not in [right_answer.lower(), wrong_answer.lower()]]

        if len(plausible_answer_array) <= 1:
            wrong_answer = select_random_word(premise) # There's still have a chance to gave "invalid" answer
            properties = """Detected span that is the SAME as the right answer, \
                                    search random word from premise"""

        else:
            wrong_answer = plausible_answer_array[0] # Take the highest value in the sorted array
            properties = """Detected span that is the SAME as the right answer, \
                                    search the highest value in the sorted array"""

        return wrong_answer, properties, plausible_answer_array


    # In[29]:


    def is_multiple_label(data, index, tag):
        
        if tag == "ner": slice='ner_tag_answer'
        elif tag == "chunking": slice='chunking_tag_answer'
        else: pass
            
        if len(data[slice][index]) > 1: return True
        else: return False


    # In[30]:


    def create_answer_match_to_multiple_label(data, 
                                            index, 
                                            tag,
                                            wrong_answer, 
                                            plausible_answer_array):
        
        if tag == "none":
            for answer in plausible_answer_array:
                if len(answer.split()) == len(wrong_answer.split()):
                    wrong_answer = answer
                    break
        
        else:
            
            if is_multiple_label(data, index, tag):
                # Check if wrong_answer a form of sentence?
                # If not, look for a wrong_answer in the form of a sentence
                if len(wrong_answer.split()) == 1:
                    for answer in plausible_answer_array:
                        if len(answer.split()) > 1:
                            wrong_answer = answer
                            break

            else:
                # Check if wrong_answer a form of word?
                # If not, look for a wrong_answer in the form of a word
                if len(wrong_answer.split()) > 1:
                    for answer in plausible_answer_array:
                        if len(answer.split()) == 1:
                            wrong_answer = answer
                            break
                            
        # We can try this out, actually.
        # With this code, we only have just check the length of each answer. 
        # We don't need is_multiple_label function check.
        #if len(right_answer) != len(wrong_answer):
        #    for answer in plausible_answer_array:
        #        if len(answer) == len(right_answer):
        #            wrong_answer = answer
        #            break
                        
        if tag == "ner": 
            properties = """IDENTICAL NER labels were found, however, \
                            the final wrong_answer is sought \
                            which is a form of a sentence"""
        
        elif tag == "chunking":
            properties = """IDENTICAL Chunking labels were found, however, \
                            the final wrong_answer is sought \
                            which is a form of a word"""
        
        elif tag == "none":
            properties = """NO CHUNKING labels were found, however, \
                            the final wrong_answer is sought \
                            which is a same form (word or sentence)"""
                        
        return wrong_answer, properties


    # In[31]:


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
                                                                        i, "ner", plausible_answer_array)
                #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "ner", wrong_answer,
                #                                                                 plausible_answer_array)
                
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
                                                                            i, "chunking", plausible_answer_array)
                    #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "chunking", wrong_answer,
                    #                                                             plausible_answer_array)
                
                # If the POS/Chunking of the right_answer cannot be detected (NULL) or context/premise 
                # does not contain any of NER of right_answer, then the final wrong_answer will be chosen 
                # based on a random word (random_word) from the context/premise.
                else:
                    for chunking_tag in chunking_tag_premise:
                        plausible_answer_array.append(chunking_tag[1])

                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "none", plausible_answer_array)
                    #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "none", wrong_answer,
                    #                                                             plausible_answer_array)

            # Check for preventing same answer for right_answer and wrong_answer  
            right_answer_span = find_substring_span(premise, right_answer)
            wrong_answer_span = find_substring_span(premise, wrong_answer)
            
            is_span_or_same_literal = check_span_overlap(right_answer_span, wrong_answer_span) \
                    or check_string_overlap(right_answer.lower(), wrong_answer.lower())

            if is_span_or_same_literal:

                # Removing right answer & wrong answer in this particular time
                wrong_answer, properties, plausible_answer_array = replace_same_answer(right_answer, 
                                                                                    wrong_answer, 
                                                                                    premise, 
                                                                                    plausible_answer_array)
                data['properties'][i] = properties
            
            data['wrong_answer'][i] = wrong_answer
            data['plausible_answer_based_on_method'][i] = plausible_answer_array
                
        return data       


    # In[32]:


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
            
            if ner_tag_answer[0][0] == "NULL":
                if data['same_chunking_tag_answer'][i] != []:
                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "chunking", plausible_answer_array)
                    #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "chunking", wrong_answer,
                    #                                                             plausible_answer_array)
                
                else:
                    for chunking_tag in chunking_tag_premise:
                        plausible_answer_array.append(chunking_tag[1])

                    wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                            i, "none", plausible_answer_array)
                    #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "none", wrong_answer,
                    #                                                             plausible_answer_array)
                continue

            # Perform NER classification
            # If the NER of the right_answer can be detected, then calculate the distance using semantic 
            # similarity or word vectors between the right_answer and various possible wrong_answers with 
            # the same NER as the right_answer. Once done, proceed to the final wrong_answer.
            if data['same_ner_tag_answer'][i] != [] and ner_tag_answer[0][0] != "NULL":
                wrong_answer, plausible_answer_array, properties = sorting_similarity(data, right_answer, \
                                                                        i, "ner", plausible_answer_array)
                #wrong_answer, properties = create_answer_match_to_multiple_label(data, i, "ner", wrong_answer,
                #                                                                 plausible_answer_array)
                
            # Kalau NER engga ada, baru drop
            # Kalau NER NULL, cek chunking
                
            # If the NER of the right_answer cannot be detected (NULL) or context/premise does not contain 
            # any of NER of right_answer, then drop that particular row data.
            else:
                data.drop(i, inplace=True)
                data.reset_index(drop=True)
                continue
            
            # Check for preventing same answer for right_answer and wrong_answer  
            right_answer_span = find_substring_span(premise, right_answer)
            wrong_answer_span = find_substring_span(premise, wrong_answer)
            
            is_span_or_same_literal = check_span_overlap(right_answer_span, wrong_answer_span) \
                    or check_string_overlap(right_answer.lower(), wrong_answer.lower())

            if is_span_or_same_literal:

                # I'm still confused, whether the overlapping 
                # answers (either in span or its literal form) 
                # should also be dropped or not.
                # If it's dropped, then, uncomment 3 lines of
                # code below.
                
                data.drop(i, inplace=True)
                data.reset_index(drop=True)
                continue

                # Removing right answer & wrong answer in this particular time
                #wrong_answer, properties, plausible_answer_array = replace_same_answer(right_answer, 
                #                                                                      wrong_answer, 
                #                                                                      premise, 
                #                                                                      plausible_answer_array)
                #data['properties'][i] = properties
            
            data['wrong_answer'][i] = wrong_answer
            data['plausible_answer_based_on_method'][i] = plausible_answer_array
                
        return data       


    # In[33]:


    data_nli_train_df = create_wrong_answer(data_nli_train_df)
    data_nli_val_df = create_wrong_answer(data_nli_val_df)
    data_nli_test_df = create_wrong_answer(data_nli_test_df)


    # # Split to two dataset: right dataset & wrong dataset

    # In[37]:


    def move_to_column_number(data, column_name="hypothesis", column_num=3):

        cols = list(data.columns)
        cols.remove(column_name)
        cols.insert(column_num, column_name)

        data = data[cols]
        
        return data


    # In[38]:


    columns_to_exclude = ['wrong_answer']

    data_nli_right_train_df = data_nli_train_df.drop(columns=columns_to_exclude).copy()
    data_nli_right_val_df = data_nli_val_df.drop(columns=columns_to_exclude).copy()
    data_nli_right_test_df = data_nli_test_df.drop(columns=columns_to_exclude).copy()


    # In[39]:


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

    # In[40]:


    def convert_question_and_answer_to_hypothesis(data):
        for i in range(len(data)):
            data['hypothesis'] = data['question'] + ' ' + data['answer']
        return data


    # In[41]:


    data_nli_right_train_df = convert_question_and_answer_to_hypothesis(data_nli_right_train_df)
    data_nli_right_val_df = convert_question_and_answer_to_hypothesis(data_nli_right_val_df)
    data_nli_right_test_df = convert_question_and_answer_to_hypothesis(data_nli_right_test_df)

    data_nli_right_train_df = move_to_column_number(data_nli_right_train_df, "hypothesis", 3)
    data_nli_right_val_df = move_to_column_number(data_nli_right_val_df, "hypothesis", 3)
    data_nli_right_test_df = move_to_column_number(data_nli_right_test_df, "hypothesis", 3)


    # In[42]:


    data_nli_wrong_train_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_train_df)
    data_nli_wrong_val_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_val_df)
    data_nli_wrong_test_df = convert_question_and_answer_to_hypothesis(data_nli_wrong_test_df)

    data_nli_wrong_train_df = move_to_column_number(data_nli_wrong_train_df, "hypothesis", 3)
    data_nli_wrong_val_df = move_to_column_number(data_nli_wrong_val_df, "hypothesis", 3)
    data_nli_wrong_test_df = move_to_column_number(data_nli_wrong_test_df, "hypothesis", 3)


    # # Add label: entailment & contradiction

    # In[43]:


    data_nli_right_train_df['label'] = 'entailment'
    data_nli_right_val_df['label'] = 'entailment'
    data_nli_right_test_df['label'] = 'entailment'

    data_nli_right_train_df = move_to_column_number(data_nli_right_train_df, "label", 4)
    data_nli_right_train_df = move_to_column_number(data_nli_right_val_df, "label", 4)
    data_nli_right_train_df = move_to_column_number(data_nli_right_test_df, "label", 4)


    # In[44]:


    data_nli_wrong_train_df['label'] = 'contradiction'
    data_nli_wrong_val_df['label'] = 'contradiction'
    data_nli_wrong_test_df['label'] = 'contradiction'

    data_nli_wrong_train_df = move_to_column_number(data_nli_wrong_train_df, "label", 4)
    data_nli_wrong_val_df = move_to_column_number(data_nli_wrong_val_df, "label", 4)
    data_nli_wrong_test_df = move_to_column_number(data_nli_wrong_test_df, "label", 4)


    # # Concat the right and wrong NLI to one NLI dataset

    # In[45]:


    data_nli_train_df_final = pd.concat([data_nli_right_train_df, data_nli_wrong_train_df], axis=0, ignore_index=True)
    data_nli_val_df_final = pd.concat([data_nli_right_val_df, data_nli_wrong_val_df], axis=0, ignore_index=True)
    data_nli_test_df_final = pd.concat([data_nli_right_test_df, data_nli_wrong_test_df], axis=0, ignore_index=True)


    # # Convert to DataFrame format to CSV

    # In[ ]:


    data_nli_train_df_final.to_csv("data_nli_train_df_keep.csv", index=False)
    data_nli_val_df_final.to_csv("data_nli_val_df_keep.csv", index=False)
    data_nli_test_df_final.to_csv("data_nli_test_df_keep.csv", index=False)


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

    print("PROGRAM FINISHED")

