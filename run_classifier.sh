#!/bin/bash

python prep_dbpedia.py
python train.py
python test.py
