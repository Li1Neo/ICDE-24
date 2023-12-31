# LAN

Code for paper "LAN: Learning Adaptive Neighbors for Real-Time ITD"

## Dataset Download

You can download dataset CERT r4.2 and r5.2 from  https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247


The expected project structure is:

```
LAN
 |-- run.sh
 |-- run.py
 |-- inference.py
 |-- model.py
 |-- utils.py
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
(Recommend)You can run `run.py` to train LAN.
You can run `inference.py` for inference

## Parameter Analysis
We analyze the influences of two key hyper-parameters of LAN, i.e., the number of candidate neighbors k in the Activity Graph Learning module and the weight of negative feedback r in the hybrid prediction loss. Regarding the performance of LAN with different numbers of candidate neighbors k obtained through retrieval, the results on the two datasets are shown in topk42.png and topk52.png. For the weight of negative feedback r in the hybrid prediction loss, the results on the two datasets are shown in weight42.png and weight52.png.


