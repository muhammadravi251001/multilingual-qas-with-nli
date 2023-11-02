import argparse
import sys
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="Program untuk fine-tuning dataset QA")
parser.add_argument('-msc', '--model_sc_name', type=str, metavar='', required=True)
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True)
parser.add_argument('-t', '--token', type=str, metavar='', required=False, default="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")
parser.add_argument('-msi', '--maximum_search_iter', type=int, metavar='', required=False, default=3)
parser.add_argument('-tq', '--type_qas', type=str, metavar='', required=False, default="entailment_or_neutral")
parser.add_argument('-ts', '--type_smoothing', type=str, metavar='', required=False, default="just_concat_answer_and_question")
parser.add_argument('-va', '--variation', type=int, metavar='', required=False, default=3)
parser.add_argument('-th', '--threshold', type=float, metavar='', required=False, default=0.5)
args = parser.parse_args()

if __name__ == "__main__":
    
    if args.data_name == "squadid": 
        MODEL_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-Squad-ID-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"
    
    elif args.data_name == "tydiqaid": 
        MODEL_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-TYDI-QA-ID-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"
    
    elif args.data_name == "idkmrc": 
        MODEL_NAME = "muhammadravi251001/fine-tuned-DatasetQAS-IDK-MRC-with-xlm-roberta-large-without-ITTL-without-freeze-LR-1e-05"

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
    
    DATA_NAME = str(args.data_name)
    HUB_TOKEN = str(args.token)
    MAXIMUM_SEARCH_ITER = int(args.maximum_search_iter)
    TYPE_QAS = str(args.type_qas)
    TYPE_SMOOTHING = str(args.type_smoothing)
    VARIATION = int(args.variation)
    THRESHOLD = float(args.threshold)

    print("Inference QA model by NLI model started!")
    print(f"Start inferencing QA model with model: xlm-roberta-large and data: {DATA_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, variation: {VARIATION}, threshold: {THRESHOLD}, and token: {HUB_TOKEN}")

    # ## Mendefinisikan hyperparameter
    HUB_TOKEN = HUB_TOKEN
    MAXIMUM_SEARCH_ITER =  MAXIMUM_SEARCH_ITER

    if HUB_TOKEN == "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": USER = "muhammadravi251001"
    
    MODEL_TG_IND_NAME = "Wikidepia/IndoT5-base-paraphrase"
    MODEL_TG_ENG_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
    MODEL_NER_NAME = "ageng-anugrah/indobert-large-p2-finetuned-ner"
    MAX_LENGTH = 512
    STRIDE = 128
    LOGGING_STEPS = 50
    WARMUP_RATIO = 0.0
    WEIGHT_DECAY = 0.0
    EVAL_STEPS_RATIO = 0.5
    SAMPLE = sys.maxsize

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    import transformers
    import evaluate
    import torch
    import operator
    import re
    import sys
    import collections
    import string
    import contextlib

    import numpy as np
    import pandas as pd
    import torch.nn as nn

    from multiprocessing import cpu_count
    from nusacrowd import NusantaraConfigHelper
    from datetime import datetime
    from tqdm import tqdm
    from deep_translator import GoogleTranslator
    from huggingface_hub import HfApi, create_repo
    
    from datasets import (
        load_dataset, 
        Dataset,
        DatasetDict
    )
    from transformers import (
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        BertForQuestionAnswering,
        AutoTokenizer,
        EarlyStoppingCallback,
        AutoModelForQuestionAnswering,
        pipeline
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

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
        
        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

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

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    def rindex(lst, value, operator=operator):
      return len(lst) - operator.indexOf(reversed(lst), value) - 1

    def preprocess_function_qa(examples, tokenizer, MAX_LENGTH=MAX_LENGTH, STRIDE=STRIDE, 
                               rindex=rindex, operator=operator):
        
        examples["question"] = [q.lstrip() for q in examples["question"]]
        examples["context"] = [c.lstrip() for c in examples["context"]]

        tokenized_examples = tokenizer(
            examples['question'],
            examples['context'],
            truncation=True,
            max_length = MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors='np'
        )
        
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []

        for seq_idx in range(len(tokenized_examples['input_ids'])):
            seq_ids = tokenized_examples.sequence_ids(seq_idx)
            offset_mappings = tokenized_examples['offset_mapping'][seq_idx]

            cur_example_idx = tokenized_examples['overflow_to_sample_mapping'][seq_idx]
            answer = examples['answer'][cur_example_idx]
            answer = eval(str(answer))
            answer_start = answer['answer_start']
            answer_end = answer['answer_end']

            context_pos_start = seq_ids.index(1)
            context_pos_end = rindex(seq_ids, 1, operator)

            s = e = 0
            if (offset_mappings[context_pos_start][0] <= answer_start and 
                offset_mappings[context_pos_end][1] >= answer_end):
                i = context_pos_start
                while offset_mappings[i][0] < answer_start:
                    i += 1
                if offset_mappings[i][0] == answer_start:
                    s = i
                else:
                    s = i - 1

                j = context_pos_end
                while offset_mappings[j][1] > answer_end:
                    j -= 1      
                if offset_mappings[j][1] == answer_end:
                    e = j
                else:
                    e = j + 1

            tokenized_examples['start_positions'].append(s)
            tokenized_examples['end_positions'].append(e)
        
        return tokenized_examples

    tokenized_data_qas_id = data_qas_id.map(
        preprocess_function_qa,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=data_qas_id['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH, 
                'STRIDE': STRIDE, 'rindex': rindex, 'operator': operator}
    )

    tokenized_data_qas_id = tokenized_data_qas_id.remove_columns(["offset_mapping", "overflow_to_sample_mapping"])
    tokenized_data_qas_id.set_format("torch", columns=["input_ids"], output_all_columns=True, device=device)
    
    tokenized_data_qas_id_train = Dataset.from_dict(tokenized_data_qas_id["train"][:SAMPLE])
    tokenized_data_qas_id_validation = Dataset.from_dict(tokenized_data_qas_id["validation"][:SAMPLE])
    tokenized_data_qas_id_test = Dataset.from_dict(tokenized_data_qas_id["test"][:SAMPLE])

    model_qa = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, num_labels=2)
    model_qa = model_qa.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        pred_tokens = normalize_text(pred).split() # True positive + False positive = For precision
        gold_tokens = normalize_text(gold).split() # True positive + False negatives = For recall
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

    def compute_metrics(predict_result, 
                    tokenized_data_qas_id_validation=tokenized_data_qas_id_validation, 
                    tokenized_data_qas_id_test=tokenized_data_qas_id_test):
    
        predictions_idx = np.argmax(predict_result.predictions, axis=2)
        denominator = len(predictions_idx[0])
        label_array = np.asarray(predict_result.label_ids)
        total_correct = 0
        f1_array = []
        
        if len(predict_result.predictions[0]) == len(tokenized_data_qas_id_validation):
            tokenized_data = tokenized_data_qas_id_validation
        
        elif len(predict_result.predictions[0]) == len(tokenized_data_qas_id_test):
            tokenized_data = tokenized_data_qas_id_test

        for i in range(len(predict_result.predictions[0])):
            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1
            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1

            pred_text = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_pred_idx: end_pred_idx])
            gold_text = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_gold_idx: end_gold_idx])

            if pred_text == gold_text:
                total_correct += 1

            f1 = compute_f1(pred=pred_text, gold=gold_text)

            f1_array.append(f1)

        exact_match = ((total_correct / denominator) * 100.0)
        final_f1 = np.mean(f1_array) * 100.0

        return {'exact_match': exact_match, 'f1': final_f1}

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
    CHECKPOINT_DIR = f'{QA}/checkpoint/'
    MODEL_DIR = f'{QA}/model/'
    OUTPUT_DIR = f'{QA}/output/'
    METRIC_RESULT_DIR = f'{QA}/metric-result/'
    REPO_NAME = f'fine-tuned-{NAME}'[:96]

    training_args_qa = TrainingArguments(
        
        # Checkpoint
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        save_strategy='steps',
        
        # Log
        report_to='tensorboard',
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=LOGGING_STEPS,
        
        # Train
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=False,
        dataloader_num_workers=cpu_count(),
        
        # Miscellaneous
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    trainer_qa = Trainer(
        model=model_qa,
        args=training_args_qa,
        tokenizer=tokenizer,
    )

    trainer_qa.save_model(MODEL_DIR)

    nlp_ner = pipeline(task="ner", model=MODEL_NER_NAME, tokenizer=MODEL_NER_NAME)

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

    predict_result = trainer_qa.predict(tokenized_data_qas_id_test)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()

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
    
    tokenizer_kwargs = {'truncation': True, 'max_length': 512}

    nlp_sc = pipeline(task="text-classification", model=MODEL_SC_NAME, tokenizer=MODEL_SC_NAME, 
                    device=torch.cuda.current_device(), **tokenizer_kwargs)

    nlp_tg_ind = pipeline(task="text2text-generation", model=MODEL_TG_IND_NAME, tokenizer=MODEL_TG_IND_NAME, 
                  device=torch.cuda.current_device(), **tokenizer_kwargs)
    
    nlp_tg_eng = pipeline(task="text2text-generation", model=MODEL_TG_ENG_NAME, tokenizer=MODEL_TG_ENG_NAME, 
                  device=torch.cuda.current_device(), **tokenizer_kwargs)
    
    def smoothing(question, pred_answer, gold_answer, type, question_word=question_word):
    
        question = question.lower()
        pred_answer = pred_answer.lower()
        gold_answer = gold_answer.lower()
        
        if type == 'replace_first':
            pred_hypothesis = question.replace('?', '')
            pred_hypothesis = pred_hypothesis.replace(question.split()[0], pred_answer)

            gold_hypothesis = question.replace('?', '')
            gold_hypothesis = gold_hypothesis.replace(question.split()[0], gold_answer)

        elif type == 'replace_question_word':
            for i in question_word:
                if i in question.split():
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = pred_hypothesis.replace(i, pred_answer)

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = gold_hypothesis.replace(i, gold_answer)
                    break
                
                else:
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = f"{pred_hypothesis.lstrip()} adalah {pred_answer}"

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = f"{gold_hypothesis.lstrip()} adalah {gold_answer}"

        elif type == 'add_adalah':
            pred_hypothesis = question.replace('?', '')
            pred_hypothesis = pred_hypothesis.replace(question.split()[0], '')
            pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

            gold_hypothesis = question.replace('?', '')
            gold_hypothesis = gold_hypothesis.replace(question.split()[0], '')
            gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"

        elif type == 'just_concat_answer_and_question':
            pred_hypothesis = f"{question} {pred_answer}"         
            gold_hypothesis = f"{question} {gold_answer}"

        elif type == 'rule_based':
            
            question = question.replace('kah', '')
            
            for j in question_word:
                
                if j in question.split():
                    
                    if j == 'siapa' or j == 'siapakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_answer} merupakan {pred_hypothesis}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_answer} merupakan {gold_hypothesis}"
                        break

                    elif j == 'apa' or j == 'apakah' or j == 'adakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"
                        break

                    elif j == 'dimana' or j == 'dimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} di {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} di {gold_answer}"
                        break

                    elif j == 'darimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} dari {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} dari {gold_answer}"
                        break

                    elif j == 'kapan' or j == 'kapankah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} pada {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} pada {gold_answer}"
                        break

                    elif j == 'bagaimana' or j == 'bagaimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '')
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"
                        break

                    elif j == 'kenapa' or j == 'mengapa':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, 'alasan').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah karena {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, 'alasan').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah karena {gold_answer}"
                        break

                    elif j == 'berapa' or j == 'berapakah' or j == 'seberapa':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()

                        if 'luas' in pred_hypothesis.split():
                            pred_hypothesis = pred_hypothesis.replace('luas', '')
                            pred_hypothesis = f"{pred_hypothesis} memiliki luas {pred_answer}"

                        elif 'jumlah' in pred_hypothesis.split():
                            pred_hypothesis = pred_hypothesis.replace('jumlah', '')
                            pred_hypothesis = f"{pred_hypothesis} berjumlah {pred_answer}"
                            
                        else: pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"
                            
                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()

                        if 'luas' in gold_hypothesis.split():
                            gold_hypothesis = gold_hypothesis.replace('luas', '')
                            gold_hypothesis = f"{gold_hypothesis} memiliki luas {gold_answer}"

                        elif 'jumlah' in gold_hypothesis.split():
                            gold_hypothesis = gold_hypothesis.replace('jumlah', '')
                            gold_hypothesis = f"{gold_hypothesis} berjumlah {gold_answer}"
                            
                        else: gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"
                        break
                    
                    else: 
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = f"{pred_hypothesis.lstrip()} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = f"{gold_hypothesis.lstrip()} adalah {gold_answer}"
                        break
                        
                else:
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = f"{pred_hypothesis.lstrip()} adalah {pred_answer}"

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = f"{gold_hypothesis.lstrip()} adalah {gold_answer}"

        elif type == 'machine_generation_with_rule_based':
            pred_hypothesis, gold_hypothesis = smoothing(question, pred_answer, gold_answer, type="rule_based")
            pred_hypothesis = nlp_tg_ind(pred_hypothesis)[0]['generated_text']
            gold_hypothesis = nlp_tg_ind(gold_hypothesis)[0]['generated_text']

        elif type == 'pure_machine_generation':
            pred_hypothesis = f"{question} {pred_answer}"         
            gold_hypothesis = f"{question} {gold_answer}"

            pred_hypothesis = nlp_tg_ind(pred_hypothesis)[0]['generated_text']
            gold_hypothesis = nlp_tg_ind(gold_hypothesis)[0]['generated_text']

        elif type == 'machine_generation_with_translation':
            pred_hypothesis, gold_hypothesis = smoothing(question, pred_answer, gold_answer, type="rule_based")

            try:
                pred_hypothesis = GoogleTranslator(source='id', target='en').translate(pred_hypothesis)
                gold_hypothesis = GoogleTranslator(source='id', target='en').translate(gold_hypothesis)

                pred_hypothesis = nlp_tg_eng(pred_hypothesis)[0]['generated_text']
                gold_hypothesis = nlp_tg_eng(gold_hypothesis)[0]['generated_text']

                pred_hypothesis = GoogleTranslator(source='en', target='id').translate(pred_hypothesis)
                gold_hypothesis = GoogleTranslator(source='en', target='id').translate(gold_hypothesis)
            
            except:
                pred_hypothesis, gold_hypothesis = smoothing(question, pred_answer, gold_answer, type="rule_based")

            if pred_hypothesis is None or gold_hypothesis is None:
                pred_hypothesis, gold_hypothesis = smoothing(question, pred_answer, gold_answer, type="rule_based")

        return pred_hypothesis.strip(), gold_hypothesis.strip()
    
    def filtering_based_on_discrete_score_nli(predict_result, 
                                          type_smoothing, 
                                          type_qas, 
                                          assign_answer_types=assign_answer_types, 
                                          MAXIMUM_SEARCH_ITER=MAXIMUM_SEARCH_ITER):
    
        predictions_idx = np.argsort(predict_result.predictions, axis=2)[:, :, 1 * -1]
        label_array = np.asarray(predict_result.label_ids)

        question_array = []
        context_array = []

        pred_answer_before_filtering_array = []
        pred_answer_after_filtering_array = []

        label_before_filtering_array = []
        label_after_filtering_array = []

        pred_hypothesis_before_filtering_array = []
        pred_hypothesis_after_filtering_array = []

        gold_answer_array = []
        gold_hypothesis_array = []

        answer_types_array = []

        for i in tqdm(range(len(predict_result.predictions[0]))):

            isFoundBiggest = False

            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1

            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1

            if len(predict_result.predictions[0]) == len(tokenized_data_qas_id_validation):
                tokenized_data = tokenized_data_qas_id_validation

            elif len(predict_result.predictions[0]) == len(tokenized_data_qas_id_test):
                tokenized_data = tokenized_data_qas_id_test

            pred_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_pred_idx: end_pred_idx], skip_special_tokens=True)

            gold_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_gold_idx: end_gold_idx], skip_special_tokens=True)

            question = []
            context = []

            start_question = tokenized_data[i]['input_ids'].index(0)
            end_question = tokenized_data[i]['input_ids'].index(2)  + 1
            start_context = end_question

            question.append(tokenized_data[i]['input_ids'][start_question: end_question])
            context.append(tokenized_data[i]['input_ids'][start_context: ])

            question_decoded = tokenizer.decode(question[0], skip_special_tokens=True)
            context_decoded = tokenizer.decode(context[0], skip_special_tokens=True)

            pred_hypothesis, gold_hypothesis = smoothing(question_decoded, pred_answer, gold_answer, type_smoothing)

            predicted_label = nlp_sc({'text': context_decoded, 
                                    'text_pair': pred_hypothesis}, 
                                    **tokenizer_kwargs)

            pred_answer_before_filtering_array.append([pred_answer])
            pred_hypothesis_before_filtering_array.append([pred_hypothesis])
            label_before_filtering_array.append([predicted_label])

            if predicted_label['label'] == 'neutral':
                if type_qas == 'entailment_or_neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])

            if predicted_label['label'] == 'entailment':
                if type_qas == 'entailment_only' or type_qas == 'entailment_or_neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])

            else:

                if predicted_label['label'] == 'neutral' and type_qas == 'entailment_or_neutral': continue

                if MAXIMUM_SEARCH_ITER < 2: 
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])
                    continue

                else:
                    pred_answer_after_filtering_array_msi_recorded = []
                    pred_hypothesis_after_filtering_array_msi_recorded = []
                    label_after_filtering_array_msi_recorded = []
                    for index_largest in range(MAXIMUM_SEARCH_ITER - 1):

                        predictions_idx_inside_loop = np.argsort(predict_result.predictions, 
                                                                axis=2)[:, :, (index_largest + 2) * -1]

                        start_pred_idx = predictions_idx_inside_loop[0][i]
                        end_pred_idx = predictions_idx_inside_loop[1][i] + 1

                        pred_answer_inside_loop = tokenizer.decode(tokenized_data[i]['input_ids']
                                                    [start_pred_idx: end_pred_idx], skip_special_tokens=True)

                        pred_hypothesis_inside_loop, gold_hypothesis = smoothing(
                            question_decoded, pred_answer_inside_loop, gold_answer, type_smoothing)

                        predicted_label_inside_loop = nlp_sc({'text': context_decoded, 
                                                            'text_pair': pred_hypothesis_inside_loop}
                                                            , **tokenizer_kwargs)

                        pred_answer_after_filtering_array_msi_recorded.append(pred_answer_inside_loop)
                        pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_inside_loop)
                        label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)

                        if type_qas == 'entailment_only':
                            if predicted_label_inside_loop['label'] == 'entailment':
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)
                                answer_types_array.append(assign_answer_types(answer=gold_answer))
                                gold_hypothesis_array.append(gold_hypothesis)

                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break

                        elif type_qas == 'entailment_or_neutral':
                            if predicted_label_inside_loop['label'] == 'entailment' or predicted_label_inside_loop['label'] == 'neutral':
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)
                                answer_types_array.append(assign_answer_types(answer=gold_answer)) 
                                gold_hypothesis_array.append(gold_hypothesis)

                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break

                    if isFoundBiggest == False:
                        
                        pred_answer_not_found_biggest = ""

                        question_array.append(question_decoded)
                        context_array.append(context_decoded)

                        pred_hypothesis_not_found_biggest, gold_hypothesis = smoothing(
                            question_decoded, pred_answer_not_found_biggest, gold_answer, type_smoothing)

                        pred_answer_after_filtering_array_msi_recorded.append(pred_answer_not_found_biggest)
                        pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_not_found_biggest)
                        label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)

                        gold_answer_array.append(gold_answer)
                        answer_types_array.append(assign_answer_types(answer=gold_answer))
                        gold_hypothesis_array.append(gold_hypothesis)

                        pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                        pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                        label_after_filtering_array.append(label_after_filtering_array_msi_recorded)

        qas_df = pd.DataFrame({'Context': context_array, 
                            'Question': question_array, 

                            'Prediction Answer Before Filtering': pred_answer_before_filtering_array,
                            'Prediction Hypothesis Before Filtering': pred_hypothesis_before_filtering_array,
                            'Label Before Filtering': label_before_filtering_array,

                            'Prediction Answer After Filtering': pred_answer_after_filtering_array,
                            'Prediction Hypothesis After Filtering': pred_hypothesis_after_filtering_array,
                            'Label After Filtering': label_after_filtering_array,

                            'Gold Answer': gold_answer_array,
                            'Gold Hypothesis': gold_hypothesis_array,

                            'Answer Type': answer_types_array,
                            'Reasoning Type': '-' 
                            })
        
        assert len(predict_result.predictions[0]) == len(qas_df)
        return qas_df
    
    def filtering_based_on_prob_dist_score_nli(predict_result, 
                                          type_smoothing, 
                                          type_qas,
                                          threshold,
                                          take_largest_prob_dist,
                                          assign_answer_types=assign_answer_types, 
                                          MAXIMUM_SEARCH_ITER=MAXIMUM_SEARCH_ITER):
    
        predictions_idx = np.argsort(predict_result.predictions, axis=2)[:, :, 1 * -1]
        label_array = np.asarray(predict_result.label_ids)

        question_array = []
        context_array = []

        pred_answer_before_filtering_array = []
        pred_answer_after_filtering_array = []

        label_before_filtering_array = []
        label_after_filtering_array = []

        pred_hypothesis_before_filtering_array = []
        pred_hypothesis_after_filtering_array = []

        gold_answer_array = []
        gold_hypothesis_array = []

        answer_types_array = []

        for i in tqdm(range(len(predict_result.predictions[0]))):

            isFoundBiggest = False

            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1

            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1

            if len(predict_result.predictions[0]) == len(tokenized_data_qas_id_validation):
                tokenized_data = tokenized_data_qas_id_validation

            elif len(predict_result.predictions[0]) == len(tokenized_data_qas_id_test):
                tokenized_data = tokenized_data_qas_id_test

            pred_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_pred_idx: end_pred_idx], skip_special_tokens=True)

            gold_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_gold_idx: end_gold_idx], skip_special_tokens=True)

            question = []
            context = []

            start_question = tokenized_data[i]['input_ids'].index(0)
            end_question = tokenized_data[i]['input_ids'].index(2)  + 1
            start_context = end_question

            question.append(tokenized_data[i]['input_ids'][start_question: end_question])
            context.append(tokenized_data[i]['input_ids'][start_context: ])

            question_decoded = tokenizer.decode(question[0], skip_special_tokens=True)
            context_decoded = tokenizer.decode(context[0], skip_special_tokens=True)


            pred_hypothesis, gold_hypothesis = smoothing(question_decoded, pred_answer, gold_answer, type_smoothing)

            predicted_label = nlp_sc({'text': context_decoded, 
                                    'text_pair': pred_hypothesis}, 
                                    **tokenizer_kwargs)

            pred_answer_before_filtering_array.append([pred_answer])
            pred_hypothesis_before_filtering_array.append([pred_hypothesis])
            label_before_filtering_array.append([predicted_label])

            if predicted_label['label'] == 'neutral' and predicted_label['score'] >= threshold:
                if type_qas == 'entailment_or_neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])

            if predicted_label['label'] == 'entailment' and predicted_label['score'] >= threshold:
                if type_qas == 'entailment_only' or type_qas == 'entailment_or_neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])

            else:

                if predicted_label['label'] == 'neutral' and predicted_label['score'] >= threshold \
                    and type_qas == 'entailment_or_neutral': continue

                if MAXIMUM_SEARCH_ITER < 2:
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    answer_types_array.append(assign_answer_types(answer=gold_answer))
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])
                    continue

                else:
                    pred_answer_after_filtering_array_msi_recorded = []
                    pred_hypothesis_after_filtering_array_msi_recorded = []
                    label_after_filtering_array_msi_recorded = []
                    for index_largest in range(MAXIMUM_SEARCH_ITER - 1):

                        #pred_answer_after_filtering_array_msi_recorded = []
                        #pred_hypothesis_after_filtering_array_msi_recorded = []
                        #label_after_filtering_array_msi_recorded = []

                        predictions_idx_inside_loop = np.argsort(predict_result.predictions, 
                                                                axis=2)[:, :, (index_largest + 2) * -1]

                        start_pred_idx = predictions_idx_inside_loop[0][i]
                        end_pred_idx = predictions_idx_inside_loop[1][i] + 1

                        pred_answer_inside_loop = tokenizer.decode(tokenized_data[i]['input_ids']
                                                    [start_pred_idx: end_pred_idx], skip_special_tokens=True)

                        pred_hypothesis_inside_loop, gold_hypothesis = smoothing(
                            question_decoded, pred_answer_inside_loop, gold_answer, type_smoothing)

                        predicted_label_inside_loop = nlp_sc({'text': context_decoded, 
                                                            'text_pair': pred_hypothesis_inside_loop}
                                                            , **tokenizer_kwargs)

                        pred_answer_after_filtering_array_msi_recorded.append(pred_answer_inside_loop)
                        pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_inside_loop)
                        label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)

                        if type_qas == 'entailment_only':
                            if predicted_label_inside_loop['label'] == 'entailment' and predicted_label_inside_loop['score'] >= threshold:
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)
                                answer_types_array.append(assign_answer_types(answer=gold_answer))
                                gold_hypothesis_array.append(gold_hypothesis)

                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break

                        elif type_qas == 'entailment_or_neutral':
                            if predicted_label_inside_loop['label'] == 'entailment' \
                                    or predicted_label_inside_loop['label'] == 'neutral' \
                                    and predicted_label_inside_loop['score'] >= threshold:
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)
                                answer_types_array.append(assign_answer_types(answer=gold_answer)) 
                                gold_hypothesis_array.append(gold_hypothesis)

                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break
                    
                    if take_largest_prob_dist == False:
                        
                        if isFoundBiggest == False:
                            
                            pred_answer_not_found_biggest = ""

                            question_array.append(question_decoded)
                            context_array.append(context_decoded)

                            pred_hypothesis_not_found_biggest, gold_hypothesis = smoothing(
                                question_decoded, pred_answer_not_found_biggest, gold_answer, type_smoothing)

                            pred_answer_after_filtering_array_msi_recorded.append(pred_answer_not_found_biggest)
                            pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_not_found_biggest)
                            label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)

                            gold_answer_array.append(gold_answer)
                            answer_types_array.append(assign_answer_types(answer=gold_answer))
                            gold_hypothesis_array.append(gold_hypothesis)

                            pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                            pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                            label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                    
                    elif take_largest_prob_dist == True:
                        
                        if isFoundBiggest == False:
                            
                            max_score = 0
                            index_max_score = 0
                            
                            for j in range(len(label_after_filtering_array[0])):
                                
                                if label_after_filtering_array[0][j]['score'] > max_score:
                                    max_score = label_after_filtering_array[0][j]['score']
                                    index_max_score = j

                            pred_answer_not_found_biggest = pred_answer_after_filtering_array_msi_recorded[index_max_score]

                            question_array.append(question_decoded)
                            context_array.append(context_decoded)

                            pred_hypothesis_not_found_biggest, gold_hypothesis = smoothing(
                                question_decoded, pred_answer_not_found_biggest, gold_answer, type_smoothing)

                            pred_answer_after_filtering_array_msi_recorded.append(pred_answer_not_found_biggest)
                            pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_not_found_biggest)
                            label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)

                            gold_answer_array.append(gold_answer)
                            answer_types_array.append(assign_answer_types(answer=gold_answer))
                            gold_hypothesis_array.append(gold_hypothesis)

                            pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                            pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                            label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                        

        qas_df = pd.DataFrame({'Context': context_array, 
                            'Question': question_array, 

                            'Prediction Answer Before Filtering': pred_answer_before_filtering_array,
                            'Prediction Hypothesis Before Filtering': pred_hypothesis_before_filtering_array,
                            'Label Before Filtering': label_before_filtering_array,

                            'Prediction Answer After Filtering': pred_answer_after_filtering_array,
                            'Prediction Hypothesis After Filtering': pred_hypothesis_after_filtering_array,
                            'Label After Filtering': label_after_filtering_array,

                            'Gold Answer': gold_answer_array,
                            'Gold Hypothesis': gold_hypothesis_array,

                            'Answer Type': answer_types_array,
                            'Reasoning Type': '-' 
                            })
        
        assert len(predict_result.predictions[0]) == len(qas_df)
        return qas_df
    
    def filtering_based_on_nli(predict_result, type_smoothing, type_qas, variation, threshold, DATA_NAME=DATA_NAME):
        
        if variation == 1:
            qas_df = filtering_based_on_discrete_score_nli(predict_result, type_smoothing, type_qas)
            
        elif variation == 2:
            qas_df = filtering_based_on_prob_dist_score_nli(predict_result, type_smoothing, 
                                                type_qas, threshold, take_largest_prob_dist=False)
        
        elif variation == 3:
            qas_df = filtering_based_on_prob_dist_score_nli(predict_result, type_smoothing, 
                                                type_qas, threshold, take_largest_prob_dist=True)
            
        return qas_df
    
    filtering_result = filtering_based_on_nli(predict_result, 
                                              type_smoothing=TYPE_SMOOTHING, 
                                              type_qas=TYPE_QAS,
                                              variation=VARIATION,
                                              threshold=THRESHOLD)

    filtering_result.to_csv(f'{OUTPUT_DIR}/output_df.csv')

    metric_result = compute_metrics(predict_result)
    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/metric_result.txt', "w") as f:
        f.write(str(metric_result))
        f.close()
    
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

    print(f"Finish inference QA with model: xlm-roberta-large and dan data: {DATA_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, variation: {VARIATION}, threshold: {THRESHOLD}, dan token: {HUB_TOKEN}")
    print("Finish inference QA model by NLI model!")