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

### Text classifier
```
Run the entire pipeline
bash run_classifier.sh
See below for breakdown
```

### Prepare Data
DBpedia dataset is parsed with bigrams and saved as `data/dbpedia_csv/train_dataset` and `data/dbpedia_csv/test_dataset`.  

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
The test script loads the trained model, and attempts to classify an example paragraph (below) as one of the 14 labels.  

Memorial Sloan Kettering Cancer Center (MSK or MSKCC) is a cancer treatment and research institution in New York City, founded in 1884 as the New York Cancer Hospital. MSKCC is the largest and oldest private cancer center in the world, and is one of 70 National Cancer Institute–designated Comprehensive Cancer Centers. Its main campus is located at 1275 York Avenue, between 67th and 68th Streets, in Manhattan.

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

### Performance on CPU vs GPU
```
Run on model (defined) above
Training set size is 532000 
Validation set size is 28000
```
```
| Epoch | CPU Time | GPU Time | Train Accuracy | Validation Accuracy |
| ----- | -------- | -------- | -------------- | ------------------- |
| 1 | 2 min 11 secs | 0 minutes, 35 secs | 96 | 98 |
| 2 | 1 min 56 secs | 0 minutes, 33 secs | 99.0 | 98.2 |
| 3 | 1 min 56 secs | 0 minutes, 32 secs | 99.6 | 98.3 |
| 4 | 1 min 56 secs | 0 minutes, 32 secs | 99.8 | 98.3 |
| 5 | 1 min 56 secs | 0 minutes, 31 secs | 99.9 | 98.4 |

The performance changes according to the size of model, dataset size etc.
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
