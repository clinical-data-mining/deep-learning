# deep-learning

## Text Classification example with PyTorch

The example follows this [tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)  

### Setup

#### Create pip virtual env
```
python3.6 -m venv env
source env/bin/activate
```

#### Download libraries
```
pip install -r requirements.txt
```

### Prepare Data
DBpedia dataset is parsed with bigrams and saved as `data/train_dataset` and `data/test_dataset`.  

To prepare the data, run
```
python prep_dbpedia.py
```

### Train Model
The training script loads the saved dataset, trains the model for 5 epochs and saves the model in `model_ep.pth`. The model is defined in `model.py`.  
For more details about the model, please see the [tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model)  
```
python train.py
```

### Test Model
The test script loads the trained model, and attempts to classify a blurb about MSKCC from wikipedia as one of the 14 labels.  
Run the `test.py` to see the results.  
```
python test.py
```
```
TextSentiment(
  (embedding): EmbeddingBag(6375026, 32, mode=mean)
  (fc): Linear(in_features=32, out_features=14, bias=True)
)
Building 14.374394416809082
EducationalInstitution 13.760090827941895
Company 12.085341453552246
Artist -0.9886159896850586
NaturalPlace -1.5990116596221924

This is a Building
```

### Notes
Check GPU usage by running:
```
nvidia-smi
```

### Next Steps
- Test performance on CPU and GPU  
    - [Make process parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)  
    - [Use multiple GPUs](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)  
