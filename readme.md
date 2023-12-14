# LAN

Code for paper "LAN: Learning Adaptive Neighbors for Real-Time ITD"

## Dataset Download

You can download dataset CERT r4.2 and r5.2 from  https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247

## Dataset Preprocess

The code for data preprocessing is located in `dataprocess.py`.
The expected project structure is:

```
LAN
 |-- run.py
 |-- inference.py
 |-- model.py
 |-- dataprocess.py
 |-- data
 |    |-- output
 |    |-- r4.2
 |    |    |-- ...  
 |    |-- r5.2
 |    |    |-- ...      
 |    |-- answers
 |    |    |-- ...  
```

## How to run
You can run `run.sh` to train LAN.
