# Experiments

## Installation and Setup

First, use virtual environment and clone this repository:
```
python -m venv multilingual-qas-with-nli-env
multilingual-qas-with-nli-env\scripts\activate
git clone https://github.com/muhammadravi251001/multilingual-qas-with-nli.git
```

After that, run experiment and install all the requirements under the current directory:
```
cd multilingual-qas-with-nli
cd experiments
pip install -r requirements.txt
```

If you not yet install git-lfs, please install. You can install git-lfs like this:
```
git lfs install
```
Or, like this:
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install
```

If you are using a new Pod via cloud, you can install all the requirements like this:
```
pip install jupyter
pip install datasets
pip install transformers
pip install tensorboard
pip install evaluate
pip install git+https://github.com/IndoNLP/nusa-crowd.git@release_exp
pip install -r requirements.txt
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install
```

Or if you had a Pod before, do it like this:
```
cd [YAML_FOLDER]" 
kubectl apply -f [POD_NAME].yaml 
kubectl get pods 
kubectl exec -it [POD_NAME] -- /bin/bash
jupyter notebook --allow-root
kubectl port-forward [POD_NAME] 8888:8888
```

## Running experiments for training Korean NLI

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_training_korean_nli.py -h
```

To run this training Korean NLI experiments, you just only do this, you optionally need to passing arguments to all parameter in `--help` menu if you don't want using the default value provided.

Example:
```
python main_training_korean_nli.py -m kykim/bert-kor-base -d kornli -e 10 -sa max
python main_training_korean_nli.py -m klue/bert-base -d kornli -e 10 -sa max
python main_training_korean_nli.py -m xlm-roberta-base -d kornli -e 10 -sa max
python main_training_korean_nli.py -m xlm-roberta-large -d kornli -e 10 -sa max
```

## Location of predictions

The predictions will be stored in `python\results\{NAME}-{TIME_NOW}`. And then this code automatically push Trainer to `{USER_that_passed_by_TOKEN}/fine-tuned-{NAME}`.