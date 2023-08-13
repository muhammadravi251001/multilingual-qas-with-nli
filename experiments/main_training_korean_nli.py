import argparse
import sys

parser = argparse.ArgumentParser(description="Training Korean NLI")
parser.add_argument('-m', '--model_name', type=str, metavar='', required=True)
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True)
parser.add_argument('-e', '--epoch', type=int, metavar='', required=True)
parser.add_argument('-sa', '--sample', type=str, metavar='', required=True)
parser.add_argument('-l', '--learn_rate', type=str, metavar='', required=False, default=1e-5)
parser.add_argument('-se', '--seed', type=int, metavar='', required=False, default=42)
parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=False, default=16)
parser.add_argument('-ga', '--gradient_accumulation', type=int, metavar='', required=False, default=8)
parser.add_argument('-t', '--token', type=str, metavar='', required=False, default="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")
args = parser.parse_args()

if __name__ == "__main__":
    
    MODEL_NAME = str(args.model_name)
    
    if "xlm-roberta" in MODEL_NAME:
        is_xlmr = True
    else:
        is_xlmr = False
    
    if (args.data_name) == "kornli":
        DATA_NAME = "KorNLI"

    if (args.sample) == "max":
        SAMPLE = sys.maxsize
    else: SAMPLE = int(args.sample)

    EPOCH = int(args.epoch)
    LEARNING_RATE = float(args.learn_rate)
    SEED = int(args.seed)
    HUB_TOKEN = str(args.token)
    BATCH_SIZE = int(args.batch_size)
    GRADIENT_ACCUMULATION = int(args.gradient_accumulation)

    if HUB_TOKEN == "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": USER = "muhammadravi251001"

    import os.path
    import wget
    
    if (os.path.exists('./multinli.train.ko.tsv') == False):
        wget.download("https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/multinli.train.ko.tsv")

    if (os.path.exists('./snli_1.0_train.ko.tsv') == False):
        wget.download("https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/snli_1.0_train.ko.tsv")

    if (os.path.exists('./xnli.dev.ko.tsv') == False):
        wget.download("https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/xnli.dev.ko.tsv")

    if (os.path.exists('./xnli.test.ko.tsv') == False):
        wget.download("https://raw.githubusercontent.com/kakaobrain/kor-nlu-datasets/master/KorNLI/xnli.test.ko.tsv")

    print(f"Start training Korean NLI with model: {MODEL_NAME}, data: {DATA_NAME}, epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, gradient_accumulation: {GRADIENT_ACCUMULATION}, and token: {HUB_TOKEN}")

    MODEL_NAME = MODEL_NAME
    EPOCH = EPOCH
    SAMPLE = SAMPLE
    LEARNING_RATE = LEARNING_RATE
    HUB_TOKEN = HUB_TOKEN
    SEED = SEED
    BATCH_SIZE = BATCH_SIZE
    GRADIENT_ACCUMULATION = GRADIENT_ACCUMULATION
    
    # You can edit this parameter, via code, not argparse
    MAX_LENGTH = 512
    STRIDE = 128
    LOGGING_STEPS = 50
    WARMUP_RATIO = 0.0
    WEIGHT_DECAY = 0.0
    EVAL_STEPS_RATIO = 0.5

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    import evaluate
    import torch
    import re

    import numpy as np
    import pandas as pd

    from multiprocessing import cpu_count
    from evaluate import load
    from datetime import datetime
    from huggingface_hub import notebook_login
    from tqdm import tqdm
    from huggingface_hub import HfApi

    from datasets import (
        load_dataset,
        Dataset,
        DatasetDict,
    )
    from transformers import (
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if (DATA_NAME == "KorNLI"):
        
        data_train_multinli = pd.read_csv("multinli.train.ko.tsv", sep='\t', on_bad_lines='skip')
        data_train_snli = pd.read_csv("snli_1.0_train.ko.tsv", sep='\t', on_bad_lines='skip')
        data_dev = pd.read_csv("xnli.dev.ko.tsv", sep='\t', on_bad_lines='skip')
        data_test = pd.read_csv("xnli.test.ko.tsv", sep='\t', on_bad_lines='skip')

        data_train = pd.concat([data_train_multinli, data_train_snli], axis=0, ignore_index=True)

        data_train = data_train[['sentence1', 'sentence2', 'gold_label']]
        data_train = data_train.rename(columns={'sentence1': 'premise', 
                                                'sentence2': 'hypothesis', 
                                                'gold_label': 'label'})

        data_dev = data_dev[['sentence1', 'sentence2', 'gold_label']]
        data_dev = data_dev.rename(columns={'sentence1': 'premise', 
                                            'sentence2': 'hypothesis', 
                                            'gold_label': 'label'})

        data_test = data_test[['sentence1', 'sentence2', 'gold_label']]
        data_test = data_test.rename(columns={'sentence1': 'premise', 
                                              'sentence2': 'hypothesis', 
                                              'gold_label': 'label'})

        data_train['label'] = data_train['label'].replace(['entailment'], 0)
        data_train['label'] = data_train['label'].replace(['contradiction'], 1)
        data_train['label'] = data_train['label'].replace(['neutral'], 2)

        data_dev['label'] = data_dev['label'].replace(['entailment'], 0)
        data_dev['label'] = data_dev['label'].replace(['contradiction'], 1)
        data_dev['label'] = data_dev['label'].replace(['neutral'], 2)

        data_test['label'] = data_test['label'].replace(['entailment'], 0)
        data_test['label'] = data_test['label'].replace(['contradiction'], 1)
        data_test['label'] = data_test['label'].replace(['neutral'], 2)

        data_train = data_train.dropna()
        data_dev = data_dev.dropna()
        data_test = data_test.dropna()

        data_train['label'] = data_train['label'].astype('int')
        data_dev['label'] = data_dev['label'].astype('int')
        data_test['label'] = data_test['label'].astype('int')

        train_dataset = Dataset.from_dict(data_train)
        dev_dataset = Dataset.from_dict(data_dev)
        test_dataset = Dataset.from_dict(data_test)

        data_nli = DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset})

    def preprocess_function_nli(examples, tokenizer, MAX_LENGTH):

        if is_xlmr:
            return tokenizer(
                examples['premise'], examples['hypothesis'],
                truncation=True,
                max_length=MAX_LENGTH
            )

        else:
            return tokenizer(
                examples['premise'], examples['hypothesis'],
                truncation=True, return_token_type_ids=True,
                max_length=MAX_LENGTH
            )

    tokenized_data_nli = data_nli.map(
        preprocess_function_nli,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=['premise', 'hypothesis'],
        fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH}
    )

    if is_xlmr:
        tokenized_data_nli.set_format("torch", columns=["input_ids"], output_all_columns=True, device=device)
    
    else:
        tokenized_data_nli.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)
    
    tokenized_data_nli_train = Dataset.from_dict(tokenized_data_nli["train"][:SAMPLE])
    tokenized_data_nli_dev = Dataset.from_dict(tokenized_data_nli["dev"][:SAMPLE])
    tokenized_data_nli_test = Dataset.from_dict(tokenized_data_nli["test"][:SAMPLE])

    id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        predictions = np.argmax(predictions, axis=1)
        labels = eval_pred.label_ids
        
        acc_result = accuracy.compute(predictions=predictions, references=labels)
        f1_result = f1.compute(predictions=predictions, references=labels, average="weighted")

        return {'accuracy': acc_result['accuracy'], 'f1': f1_result['f1']}

    model_sc = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, 
        id2label=id2label, label2id=label2id)

    model_sc = model_sc.to(device)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
    
    if (re.findall(r'.*/(.*)$', MODEL_NAME) == []): 
        NAME = f'KoreanNLI-{DATA_NAME}-with-{str(MODEL_NAME)}'
    else:
        new_name = re.findall(r'.*/(.*)$', MODEL_NAME)[0]
        NAME = f'KoreanNLI-{DATA_NAME}-with-{str(new_name)}'
    
    SC = f'./results/{NAME}-{TIME_NOW}'
    CHECKPOINT_DIR = f'{SC}/checkpoint/'
    MODEL_DIR = f'{SC}/model/'
    OUTPUT_DIR = f'{SC}/output/'
    METRIC_RESULT_DIR = f'{SC}/metric-result/'
    REPO_NAME = f'fine-tuned-{NAME}'[:96]

    training_args_sc = TrainingArguments(
        
        # Checkpoint
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        save_strategy='steps',
        save_total_limit=EPOCH,
        
        # Log
        report_to='tensorboard',
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=LOGGING_STEPS,
        
        # Train
        num_train_epochs=EPOCH,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=False,
        dataloader_num_workers=cpu_count(),
        
        # Miscellaneous
        evaluation_strategy='steps',
        save_steps=int((data_nli['train'].num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        eval_steps=int((data_nli['train'].num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        seed=SEED,
        hub_token=HUB_TOKEN,
        push_to_hub=True,
        hub_model_id=REPO_NAME,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    trainer_sc = Trainer(
        model=model_sc,
        args=training_args_sc,
        train_dataset=tokenized_data_nli_train,
        eval_dataset=tokenized_data_nli_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer_sc.train()

    trainer_sc.save_model(MODEL_DIR)

    def represent_prediction_output(predict_result):
        predictions_idx = np.argmax(predict_result.predictions, axis=1)
        label_array = np.asarray(predict_result.label_ids)

        premise_array = []
        hypothesis_array = []

        pred_label_array = []
        gold_label_array = []

        for i in tqdm(range(len(predict_result.predictions))):

            premise = []
            hypothesis = []

            if len(predict_result.predictions) == len(tokenized_data_nli_dev):
                tokenized_data = tokenized_data_nli_dev

            elif len(predict_result.predictions) == len(tokenized_data_nli_test):
                tokenized_data = tokenized_data_nli_test

            if is_xlmr:
                start_premise = tokenized_data[i]['input_ids'].index(0)
                end_premise = tokenized_data[i]['input_ids'].index(2)  + 1
                start_hypothesis = end_premise

                premise.append(tokenized_data[i]['input_ids'][start_premise: end_premise])
                hypothesis.append(tokenized_data[i]['input_ids'][start_hypothesis: ])

                premise_decoded = tokenizer.decode(premise[0], skip_special_tokens=True)
                hypothesis_decoded = tokenizer.decode(hypothesis[0], skip_special_tokens=True)

            else:
                for j in range(len(tokenized_data[i]['token_type_ids'])):
                    if tokenized_data_nli_test[i]['token_type_ids'][j] == 0:
                        premise.append(tokenized_data[i]['input_ids'][j])

                    else:
                        hypothesis.append(tokenized_data[i]['input_ids'][j])

                    premise_decoded = tokenizer.decode(premise, skip_special_tokens=True)
                    hypothesis_decoded = tokenizer.decode(hypothesis, skip_special_tokens=True)

            premise_array.append(premise_decoded)
            hypothesis_array.append(hypothesis_decoded)

            pred_label_array.append(predictions_idx[i])
            gold_label_array.append(label_array[i])

        id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        nli_df = pd.DataFrame({'Premise': premise_array, 
                                'Hypothesis': hypothesis_array,
                                'Prediction Label': pred_label_array,
                                'Gold Label': gold_label_array
                                })

        nli_df["Prediction Label"] = nli_df["Prediction Label"].map(id2label)
        nli_df["Gold Label"] = nli_df["Gold Label"].map(id2label)

        return nli_df
    
    predict_result = trainer_sc.predict(tokenized_data_nli_test)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()

    nli_df = represent_prediction_output(predict_result)
    nli_df.to_csv(f'{OUTPUT_DIR}/output_df.csv')
    
    metric_result = compute_metrics(predict_result)
    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/metric_result.txt', "w") as f:
        f.write(str(metric_result))
        f.close()

    api = HfApi()

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

    print(f"Finish training Korean NLI with model: {MODEL_NAME}, data: {DATA_NAME}, epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, gradient_accumulation: {GRADIENT_ACCUMULATION}, and token: {HUB_TOKEN}")
    print("Finish Training Korean NLI!")