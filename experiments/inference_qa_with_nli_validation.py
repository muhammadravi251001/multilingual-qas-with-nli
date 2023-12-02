import argparse
import sys
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="Inference QA with NLI validation")
parser.add_argument('-msc', '--model_sc_name', type=str, metavar='', required=True)
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True)
parser.add_argument('-t', '--token', type=str, metavar='', required=False, default="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")
parser.add_argument('-msi', '--maximum_search_iter', type=int, metavar='', required=False, default=3)
parser.add_argument('-tq', '--type_qas', type=str, metavar='', required=False, default="entailment_or_neutral")
parser.add_argument('-ts', '--type_smoothing', type=str, metavar='', required=False, default="just_concat_answer_and_question")
parser.add_argument('-va', '--variation', type=int, metavar='', required=False, default=1)
parser.add_argument('-th', '--threshold', type=float, metavar='', required=False, default=False)
args = parser.parse_args()

if __name__ == "__main__":

    MODEL_SC_NAME = str(args.model_sc_name)
    DATA_NAME = str(args.data_name)
    HUB_TOKEN = str(args.token) 
    MAXIMUM_SEARCH_ITER = int(args.maximum_search_iter)
    TYPE_QAS = str(args.type_qas)
    TYPE_SMOOTHING = str(args.type_smoothing)
    VARIATION = int(args.variation)
    THRESHOLD = float(args.threshold)
    
    if HUB_TOKEN == "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": USER = "muhammadravi251001"  
    MAX_LENGTH = 512 
    MODEL_TG_IND_NAME = "Wikidepia/IndoT5-base-paraphrase"
    MODEL_TG_ENG_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
    MODEL_NER_NAME = "ageng-anugrah/indobert-large-p2-finetuned-ner"
    NO_ANSWER_STATEMENT = "Tidak ada jawaban"

    if args.data_name == "squadid": 
        MODEL_QA_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-Squad-ID-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"
    
    elif args.data_name == "tydiqaid": 
        MODEL_QA_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-TYDI-QA-ID-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"
    
    elif args.data_name == "idkmrc": 
        MODEL_QA_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-IDK-MRC-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"

    if args.model_sc_name == "indonli":
        MODEL_SC_NAME = "muhammadravi251001/fine-tuned-NLI-indonli-with-xlm-roberta-large"
    
    elif args.model_sc_name == "indonli_mnli":
        MODEL_SC_NAME = "muhammadravi251001/fine-tuned-NLI-indonli_mnli-with-xlm-roberta-large"
    
    elif args.model_sc_name == "indonli_mnli_idkmrc-nli":
        MODEL_SC_NAME = "muhammadravi251001/fine-tuned-NLI-indonli_mnli_idkmrc-nli-with-xlm-roberta-large" 
    
    elif args.model_sc_name == "indonli_mnli_tydiqaid-nli":
        MODEL_SC_NAME = "muhammadravi251001/fine-tuned-NLI-indonli_mnli_tydiqaid-nli-with-xlm-roberta-large"
    
    elif args.model_sc_name == "indonli_mnli_squadid-nli":
        MODEL_SC_NAME = "muhammadravi251001/fine-tuned-NLI-indonli_mnli_squadid-nli-with-xlm-roberta-large"
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    import torch
    import re
    import sys
    import collections
    import string

    import numpy as np
    import pandas as pd

    from multiprocessing import cpu_count
    from nusacrowd import NusantaraConfigHelper
    from datetime import datetime
    from tqdm import tqdm
    from huggingface_hub import HfApi, create_repo
    from deep_translator import GoogleTranslator

    from datasets import (
        load_dataset, 
        Dataset,
        DatasetDict
    )
    from transformers import (
        pipeline,
        AutoModelForQuestionAnswering, 
        AutoTokenizer, 
        AutoModelForTokenClassification, 
        AutoModelForSequenceClassification, 
        T5ForConditionalGeneration, 
        T5Tokenizer
    )

    print("Inference QA model by NLI model started!")
    print(f"Start inferencing QA model with model: {args.model_sc_name} and data: {DATA_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, variation: {VARIATION}, threshold: {THRESHOLD}, and token: {HUB_TOKEN}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (DATA_NAME == "squadid"):
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

    elif (DATA_NAME == "idkmrc"):
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

    elif (DATA_NAME == "tydiqaid"):
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

    test_data = pd.DataFrame(data_qas['test'])
    test_data

    tokenizer_kwargs = {'truncation': True, 'max_length': 512}

    tokenizer_qa = AutoTokenizer.from_pretrained(MODEL_QA_NAME)
    model_qa = AutoModelForQuestionAnswering.from_pretrained(MODEL_QA_NAME)
    model_qa.config.top_k = MAXIMUM_SEARCH_ITER
    model_qa = model_qa.to(device)

    tokenizer_ner = AutoTokenizer.from_pretrained(MODEL_NER_NAME)
    model_ner = AutoModelForTokenClassification.from_pretrained(MODEL_NER_NAME)
    model_ner = model_ner.to(device)

    tokenizer_sc = AutoTokenizer.from_pretrained(MODEL_SC_NAME)
    model_sc = AutoModelForSequenceClassification.from_pretrained(MODEL_SC_NAME)
    model_sc = model_sc.to(device)

    tokenizer_tg_ind = T5Tokenizer.from_pretrained(MODEL_TG_IND_NAME)
    model_tg_ind = T5ForConditionalGeneration.from_pretrained(MODEL_TG_IND_NAME)
    model_tg_ind = model_tg_ind.to(device)

    tokenizer_tg_eng = T5Tokenizer.from_pretrained(MODEL_TG_ENG_NAME)
    model_tg_eng = T5ForConditionalGeneration.from_pretrained(MODEL_TG_ENG_NAME)
    model_tg_eng = model_tg_eng.to(device)

    """ Uncomment this if you want to use pipeline instead of .predict()
    nlp_qa = pipeline(task="question-answering", model=MODEL_QA_NAME, tokenizer=MODEL_QA_NAME, device=torch.cuda.current_device(), **tokenizer_kwargs)
    nlp_ner = pipeline(task="ner", model=MODEL_NER_NAME, tokenizer=MODEL_NER_NAME, device=torch.cuda.current_device())
    nlp_sc = pipeline(task="text-classification", model=MODEL_SC_NAME, tokenizer=MODEL_SC_NAME, device=torch.cuda.current_device(), **tokenizer_kwargs)
    nlp_tg_ind = pipeline(task="text2text-generation", model=MODEL_TG_IND_NAME, tokenizer=MODEL_TG_IND_NAME, device=torch.cuda.current_device(), **tokenizer_kwargs)
    nlp_tg_eng = pipeline(task="text2text-generation", model=MODEL_TG_ENG_NAME, tokenizer=MODEL_TG_ENG_NAME, device=torch.cuda.current_device(), **tokenizer_kwargs)
    """
    
    def nlp_qa(question, context, top_k=MAXIMUM_SEARCH_ITER):
        
        inputs = tokenizer_qa(question, context, 
                              return_tensors="pt",
                              **tokenizer_kwargs)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model_qa(**inputs)
        
        sorted_start_logits = torch.argsort(outputs.start_logits)
        sorted_end_logits = torch.argsort(outputs.end_logits)

        answer_array = []
        for i in range(1, (top_k + 1)):
            
            start_index = sorted_start_logits[0, -i]
            end_index = sorted_end_logits[0, -i]
            answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]

            answer = tokenizer_qa.decode(answer_tokens)
            answer_array.append({'answer': answer})
        
        return answer_array
    
    def nlp_sc(text_dict):
    
        inputs = tokenizer_sc(text_dict['text'], text_dict['text_pair'], 
                              return_tensors="pt",
                              **tokenizer_kwargs)
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model_sc(**inputs)

        label_id = torch.argmax(outputs.logits).item()
        label = model_sc.config.id2label[label_id]
        score = outputs.logits.softmax(dim=-1)[0][label_id].item()

        return {'label': label, 'score': score}
    
    def nlp_ner(sentence):

        inputs = tokenizer_ner(sentence,
                        return_offsets_mapping=True,
                        return_tensors="pt",
                        **tokenizer_kwargs)
        
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        outputs = model_ner(ids, attention_mask=mask)
        logits = outputs.logits

        active_logits = logits.view(-1, model_ner.config.num_labels)
        flattened_predictions = torch.argmax(active_logits, dim=1)

        tokens = tokenizer_ner.tokenize(sentence)
        token_predictions = [model_ner.config.id2label[i] for i in flattened_predictions.cpu().numpy()]

        offset_mapping = inputs["offset_mapping"].squeeze().tolist()

        results = []

        entity = None
        start_index = 0

        for i, (token, token_pred, mapping) in enumerate(zip(tokens, token_predictions, offset_mapping)):

            if entity and entity != 'O':
                results.append({
                    'entity': entity,
                    'score': 1.0,
                    'index': i - 1,
                    'word': tokens[start_index:i][0],
                    'start': start_index,
                    'end': i - 1
                })
            entity = token_pred
            start_index = i

        return results

    def nlp_tg_ind(prompt):
        
        input_ids = tokenizer_tg_ind(prompt, return_tensors="pt").input_ids.to(device)
        
        output = model_tg_ind.generate(input_ids,
                                       num_return_sequences=1, 
                                       no_repeat_ngram_size=2, 
                                       top_k=50, 
                                       top_p=0.95,
                                       **tokenizer_kwargs)
        
        generated_text = tokenizer_tg_ind.batch_decode(output, skip_special_tokens=True)
        
        return [{'generated_text': generated_text[0]}]
    
    def nlp_tg_eng(prompt):
        
        input_ids = tokenizer_tg_eng(prompt, return_tensors="pt").input_ids.to(device)
        
        output = model_tg_eng.generate(input_ids,
                                       num_return_sequences=1, 
                                       no_repeat_ngram_size=2, 
                                       top_k=50, 
                                       top_p=0.95,
                                       **tokenizer_kwargs)
        
        generated_text = tokenizer_tg_eng.batch_decode(output, skip_special_tokens=True)
        
        return [{'generated_text': generated_text[0]}]
    
    def retrieve_answer_text(data):
        for i in range(len(data)):
            data['answer'][i] = data['answer'][i]['text']
        return data

    test_data = retrieve_answer_text(test_data)

    def assign_answer_types(answer, nlp=nlp_ner):

        if answer == str(): 
            return ["NULL"]

        entity_array = []    
        ner_result = nlp(answer)

        for i in ner_result:
            entity = i['entity'][2:]
            entity_array.append(entity)

        if entity_array == []: 
            return ["NULL"]

        return list(set(entity_array))

    question_word = ['siapa', 'siapakah',
                    'apa', 'apakah', 'adakah',
                    'dimana', 'dimanakah', 'darimanakah',
                    'kapan', 'kapankah',
                    'bagaimana', 'bagaimanakah',
                    'kenapa', 'mengapa',
                    'berapa', 'berapakah', 'seberapa',
                
                    'Siapa', 'Siapakah',
                    'Apa', 'Apakah', 'Adakah',
                    'Dimana', 'Dimanakah', 'Darimanakah',
                    'Kapan', 'Kapankah',
                    'Bagaimana', 'Bagaimanakah',
                    'Kenapa', 'Mengapa',
                    'Berapa', 'Berapakah', 'Seberapa'
                    ]
    
    def smoothing(question, answer, type):

        if type == 'replace_first':
            hypothesis = question.replace('?', '')
            hypothesis = hypothesis.replace(question.split()[0], answer)

        elif type == 'replace_question_word':
            
            for i in question_word:
                
                if i in question.split():
                    hypothesis = question.replace('?', '')
                    hypothesis = hypothesis.replace(i, answer)
                    break
                
                else:
                    hypothesis = question.replace('?', '')
                    hypothesis = f"{hypothesis.lstrip()} adalah {answer}"

        elif type == 'add_adalah':
            hypothesis = question.replace('?', '')
            hypothesis = hypothesis.replace(question.split()[0], '')
            hypothesis = f"{hypothesis} adalah {answer}"

        elif type == 'rule_based':
            
            question = question.replace('kah', '')
            
            for j in question_word:
                
                if j in question.split():
                    
                    if j == 'siapa' or j == 'siapakah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()
                        hypothesis = f"{answer} merupakan {hypothesis}"
                        break

                    elif j == 'apa' or j == 'apakah' or j == 'adakah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()
                        hypothesis = f"{hypothesis} adalah {answer}"
                        break

                    elif j == 'dimana' or j == 'dimanakah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()
                        hypothesis = f"{hypothesis} di {answer}"
                        break

                    elif j == 'darimanakah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()
                        hypothesis = f"{hypothesis} dari {answer}"
                        break

                    elif j == 'kapan' or j == 'kapankah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()
                        hypothesis = f"{hypothesis} pada {answer}"
                        break

                    elif j == 'bagaimana' or j == 'bagaimanakah':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '')
                        hypothesis = f"{hypothesis} adalah {answer}"
                        break

                    elif j == 'kenapa' or j == 'mengapa':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, 'alasan').lstrip()
                        hypothesis = f"{hypothesis} adalah karena {answer}"
                        break

                    elif j == 'berapa' or j == 'berapakah' or j == 'seberapa':
                        hypothesis = question.replace('?', '')
                        hypothesis = hypothesis.replace(j, '').lstrip()

                        if 'luas' in hypothesis.split():
                            hypothesis = hypothesis.replace('luas', '')
                            hypothesis = f"{hypothesis} memiliki luas {answer}"

                        elif 'jumlah' in hypothesis.split():
                            hypothesis = hypothesis.replace('jumlah', '')
                            hypothesis = f"{hypothesis} berjumlah {answer}"
                            
                        else: hypothesis = f"{hypothesis} adalah {answer}"
                            
                        break
                    
                    else: 
                        hypothesis = question.replace('?', '')
                        hypothesis = f"{hypothesis.lstrip()} adalah {answer}"
                        break
                        
                else:
                    hypothesis = question.replace('?', '')
                    hypothesis = f"{hypothesis.lstrip()} adalah {answer}"

        elif type == 'machine_generation_with_rule_based':
            hypothesis = smoothing(question, answer, type="rule_based")
            hypothesis = nlp_tg_ind(hypothesis)[0]['generated_text']

        elif type == 'pure_machine_generation':
            hypothesis = f"{question} {answer}"         
            hypothesis = nlp_tg_ind(hypothesis)[0]['generated_text']

        elif type == 'machine_generation_with_translation':
            hypothesis = smoothing(question, answer, type="rule_based")

            try:
                hypothesis = GoogleTranslator(source='id', target='en').translate(hypothesis)
                hypothesis = nlp_tg_eng(hypothesis)[0]['generated_text']
                hypothesis = GoogleTranslator(source='en', target='id').translate(hypothesis)
            
            except:
                hypothesis = smoothing(question, answer, type="rule_based")

            if hypothesis is None:
                hypothesis = smoothing(question, answer, type="rule_based")
        
        elif type == 'just_concat_answer_and_question':

            #if answer != "" and answer == str:
            #    hypothesis = f"{question} {answer}"

            #else:
            #    hypothesis = f"{question} {NO_ANSWER_STATEMENT}"

            hypothesis = f"{question} {answer}"

        return hypothesis.strip()

    def create_df_with_prediction(df):
        
        def retrieve_answer_text_from_list(arr):
            
            if type(arr) != list:
                arr = [arr]
            
            answer_text_array = []
            for i in range(len(arr)):
                answer_text_array.append(arr[i]['answer'])
            
            assert type(answer_text_array) == list
            return answer_text_array
        
        def smoothing_from_list(pred_answer_arr, question):
            
            hypothesis_array = []
            for answer in pred_answer_arr:
                hypothesis_array.append(smoothing(question, answer, TYPE_SMOOTHING))
                
            assert type(hypothesis_array) == list
            return hypothesis_array
        
        def retrieve_label_from_list(pred_hypothesis_arr, context):
            
            label_array = []
            for hypothesis in pred_hypothesis_arr:
                pred_label = nlp_sc({'text': context, 'text_pair': hypothesis})
                label_array.append(pred_label)
            assert type(label_array) == list
            return label_array
        
        pred_answer_array = []
        pred_hypothesis_array = []
        pred_label_array = []
        
        gold_hypothesis_array = []
        answer_types_array = []
        
        for i in tqdm(range(len(df))):
            
            context = df['context'][i]
            question = df['question'][i]
            gold_answer = df['answer'][i]
            
            answer_types_array.append(assign_answer_types(answer=gold_answer))
            gold_hypothesis_array.append(smoothing(question, gold_answer, TYPE_SMOOTHING))
            
            pred_answer = retrieve_answer_text_from_list(nlp_qa(context=context, 
                                                                question=question, 
                                                                top_k=MAXIMUM_SEARCH_ITER))
            
            pred_hypothesis = smoothing_from_list(pred_answer, question)
            pred_label = retrieve_label_from_list(pred_hypothesis, context)
            
            pred_answer_array.append(pred_answer)
            pred_hypothesis_array.append(pred_hypothesis)
            pred_label_array.append(pred_label)
            
        df = pd.DataFrame({
                            'Context': df['context'], 
                            'Question': df['question'], 

                            'Rec. Pred Answer': pred_answer_array,
                            'Rec. Pred Hypothesis': pred_hypothesis_array,
                            'Rec. Pred Label': pred_label_array,

                            'Gold Answer': df ['answer'],
                            'Gold Hypothesis': gold_hypothesis_array,

                            'Gold Answer Type': answer_types_array,
                            'Reasoning Type': '-',
                            
                            'Properties': '-',
                            })
        
        return df

    qas_df = create_df_with_prediction(test_data)
    qas_df

    def create_df_with_final_answer_by_variation(df, type_qas=TYPE_QAS, variation=1, threshold=None):
        
        def search_final_answer_by_variation(label, type_qas=type_qas, variation=variation, threshold=threshold):
            
            if variation == 1:
                
                if type_qas == 'entailment_only':

                    for i, prediction in enumerate(label):
                        if prediction['label'] == 'entailment':
                            return i
                    return None

                elif type_qas == 'entailment_or_neutral':

                    for i, prediction in enumerate(label):
                        if prediction['label'] == 'entailment' or prediction['label'] == 'neutral':
                            return i
                    return None
            
            elif (variation == 2 or variation == 3) and threshold > 0:
                
                if type_qas == 'entailment_only':

                    for i, prediction in enumerate(label):
                        if prediction['label'] == 'entailment' and prediction['score'] >= threshold:
                            return i
                    return None

                elif type_qas == 'entailment_or_neutral':

                    for i, prediction in enumerate(label):
                        if (prediction['label'] == 'entailment' or prediction['label'] == 'neutral')\
                            and prediction['score'] >= threshold:
                            return i
                    return None

        def take_highest_prob(label):
    
            entailments = [item for item in label if item['label'] == 'entailment']
            if entailments:
                best_entailment = max(entailments, key=lambda x: x['score'])
                pred_answer_index = label.index(best_entailment)
                return pred_answer_index
                
            neutrals = [item for item in label if item['label'] == 'neutral']
            if neutrals:
                best_neutral = max(neutrals, key=lambda x: x['score'])
                pred_answer_index = label.index(best_neutral)
                return pred_answer_index
                
            contradictions = [item for item in label if item['label'] == 'contradiction']
            if contradictions:
                smallest_contradiction = min(contradictions, key=lambda x: x['score'])
                pred_answer_index = label.index(smallest_contradiction)
                return pred_answer_index

        pred_answer_array = []
        pred_hypothesis_array = []
        pred_label_array = []
        properties_array = []
        
        for i in range(len(df)):

            if variation == 0:

                properties = "Answer saved from variation 0 with same as baseline model do"
                pred_answer = df['Rec. Pred Answer'][i][0]
                pred_label = df['Rec. Pred Label'][i][0]
            
            if variation == 1:
                
                index = search_final_answer_by_variation(df['Rec. Pred Label'][i], variation=1, threshold=threshold)
                
                if index is not None:
                    properties = "Answer saved from variation 1 with founded index"
                    pred_answer = df['Rec. Pred Answer'][i][index]
                    pred_label = df['Rec. Pred Label'][i][index]
                
                else:
                    properties = "(Blank) answer saved from variation 1 without founded index"
                    pred_answer = ""
                    pred_label = ""
            
            elif variation == 2 and threshold:
                
                index = search_final_answer_by_variation(df['Rec. Pred Label'][i], variation=2, threshold=threshold)
                
                if index is not None:
                    properties = "Answer saved from variation 2 with founded index"
                    pred_answer = df['Rec. Pred Answer'][i][index]
                    pred_label = df['Rec. Pred Label'][i][index]
                
                else:
                    properties = "(Blank) answer saved from variation 2 without founded index"
                    pred_answer = ""
                    pred_label = ""
            
            elif variation == 3 and threshold:
                
                index = search_final_answer_by_variation(df['Rec. Pred Label'][i], variation=3, threshold=threshold)
                
                if index is not None:
                    properties = "Answer saved from variation 3 with founded index"
                    pred_answer = df['Rec. Pred Answer'][i][index]
                    pred_label = df['Rec. Pred Label'][i][index]
                
                else:           
                    properties = "Highest probability answer saved from variation 3 without founded index"
                    highest_prob_idx = take_highest_prob(df['Rec. Pred Label'][i])
                    pred_answer = df['Rec. Pred Answer'][i][highest_prob_idx]
                    pred_label = df['Rec. Pred Label'][i][highest_prob_idx]
                    
            pred_hypothesis = smoothing(df['Question'][i], pred_answer, TYPE_SMOOTHING)
                
            pred_answer_array.append(pred_answer)
            pred_hypothesis_array.append(pred_hypothesis)
            pred_label_array.append(pred_label)
            properties_array.append(properties)
            
        df = pd.DataFrame({
                            'Context': df['Context'], 
                            'Question': df['Question'], 
            
                            'Prediction Answer': pred_answer_array,
                            'Prediction Hypothesis': pred_hypothesis_array,
                            'Prediction Label': pred_label_array,

                            'Rec. Pred Answer': df['Rec. Pred Answer'],
                            'Rec. Pred Hypothesis': df['Rec. Pred Hypothesis'],
                            'Rec. Pred Label': df['Rec. Pred Label'],

                            'Gold Answer': df['Gold Answer'],
                            'Gold Hypothesis': df['Gold Hypothesis'],

                            'Gold Answer Type': df['Gold Answer Type'],
                            'Reasoning Type': df['Reasoning Type'],
                            
                            'Properties': properties_array,
                            })
        
        return df

    qas_df = create_df_with_final_answer_by_variation(qas_df, type_qas=TYPE_QAS, variation=VARIATION, threshold=THRESHOLD)
    qas_df

    def normalize_text(s):
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(pred, gold):
        pred_tokens = normalize_text(pred).split() # True positive + False positive = Untuk precision
        gold_tokens = normalize_text(gold).split() # True positive + False negatives = Untuk recall
        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values()) # True positive

        if len(gold_tokens) == 0 or len(pred_tokens) == 0: 
            return int(gold_tokens == pred_tokens)

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)

        return f1

    def compute_metrics_from_df(df):
        
        denominator = len(df)
        total_correct = 0
        f1_array = []

        for i in range(len(df)):

            pred_answer = df["Prediction Answer"][i]
            gold_answer = df["Gold Answer"][i]

            if pred_answer == gold_answer:
                total_correct += 1

            f1 = compute_f1(pred=pred_answer, gold=gold_answer)

            f1_array.append(f1)
            
        exact_match = ((total_correct / denominator) * 100.0)
        final_f1 = np.mean(f1_array) * 100.0

        return {'exact_match': exact_match, 'f1': final_f1}

    metric_result = compute_metrics_from_df(qas_df)

    TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")

    if TYPE_QAS == "entailment_only": TQ_CODE = "TQ1"
    elif TYPE_QAS == "entailment_or_neutral": TQ_CODE = "TQ2"

    if TYPE_SMOOTHING == "replace_first": TS_CODE = "TS1"
    elif TYPE_SMOOTHING == "replace_question_word": TS_CODE = "TS2"
    elif TYPE_SMOOTHING == "add_adalah": TS_CODE = "TS3"
    elif TYPE_SMOOTHING == "just_concat_answer_and_question": TS_CODE = "TS4"
    elif TYPE_SMOOTHING == "rule_based": TS_CODE = "TS5"
    elif TYPE_SMOOTHING == "machine_generation_with_rule_based": TS_CODE = "TS6"
    elif TYPE_SMOOTHING == "pure_machine_generation": TS_CODE = "TS7"
    elif TYPE_SMOOTHING == "machine_generation_with_translation": TS_CODE = "TS8"

    MSI_CODE = f"MS{MAXIMUM_SEARCH_ITER}"
    VARIATION_CODE = f"VA{VARIATION}"
    THRESHOLD_CODE = f"TH{THRESHOLD}"

    NAME = f'FilteringNLI-{args.model_sc_name}-{args.data_name}-{TQ_CODE}-{TS_CODE}-{MSI_CODE}-{VARIATION_CODE}-{THRESHOLD_CODE}'

    QA = f'./results/{NAME}-{TIME_NOW}'
    CHECKPOINT_DIR = f'{QA}/checkpoint'
    MODEL_DIR = f'{QA}/model'
    OUTPUT_DIR = f'{QA}/output'
    METRIC_RESULT_DIR = f'{QA}/metric-result'
    REPO_NAME = f'fine-tuned-{NAME}'[:96]
    
    if not os.path.exists(QA):
        os.makedirs(QA)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(METRIC_RESULT_DIR):
        os.makedirs(METRIC_RESULT_DIR)
    
    with open(f'{METRIC_RESULT_DIR}/metric_result.txt', "w") as f:
        f.write(str(metric_result))
        f.close()

    qas_df.to_csv(f'{OUTPUT_DIR}/output_df.csv')

    api = HfApi()

    try:
        
        api.upload_folder(
            folder_path=f"{OUTPUT_DIR}",
            repo_id=f"{USER}/{REPO_NAME}",
            repo_type="model",
            token=HUB_TOKEN,
            path_in_repo="results/output",
        )

        api.upload_folder(
            folder_path=f"{METRIC_RESULT_DIR}",
            repo_id=f"{USER}/{REPO_NAME}",
            repo_type="model",
            token=HUB_TOKEN,
            path_in_repo="results/evaluation",
        )
    
    except:

        create_repo(f"{USER}/{REPO_NAME}", token=HUB_TOKEN)
        
        api.upload_folder(
            folder_path=f"{OUTPUT_DIR}",
            repo_id=f"{USER}/{REPO_NAME}",
            repo_type="model",
            token=HUB_TOKEN,
            path_in_repo="results/output",
        )

        api.upload_folder(
            folder_path=f"{METRIC_RESULT_DIR}",
            repo_id=f"{USER}/{REPO_NAME}",
            repo_type="model",
            token=HUB_TOKEN,
            path_in_repo="results/evaluation",
        )

    print(f"Finish inference QA with model: {args.model_sc_name} and dan data: {DATA_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, variation: {VARIATION}, threshold: {THRESHOLD}, dan token: {HUB_TOKEN}")
    print("Finish inference QA model by NLI model!")