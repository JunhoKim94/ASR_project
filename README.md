# ASR_project
This repository created for ASR Hackathon Competition. 

We won the Excellence award in NHN Consortium 2020 AI Training Data Hackathon Competition
http://hackathon.workpedia.co.kr/

## Requirements
```
pip install chainer
pip install hgtk
pip install python-Levenshtein
pip install typeguard
pip install librosa
pip install configargparse
pip install torch_complex
pip install pytorch_wpe
pip install humanfriendly

conda install editdistance
```

## Total progress

### 1) Preprocess the dataset

![Preprocess](./images/preprocess.JPG)
### 2) Models

#### Encoding

![Convolutional Neural Network](./images/CNN.JPG)

#### Transformer model

![Transformers](./images/Transformers.JPG)

### 3) Hyper Parameter Optimize

We used bayesian optimization to find optimal beam size, penalty score, and CTC weight to inference model

![Bayesian Optimization](./images/BayesianOptimization.JPG)

## Results

We achieve the 4.5 CER in NHN ASR hackerthon dataset (not publicly available now)

## Usage

Input data folder samples are in data folder.
```
#input_data_path : folder
#output_data_path : txt file
python evaluation --input_dir "input_data_path" --output_dir "output_data_path"
```
![Usage](./images/usage.JPG)
