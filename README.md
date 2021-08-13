# RPLP
Relation Projection for Link Prediction: Bridging Message Passing and Triplet Convolution

This repository contains all the source code of the implementation for the paper: Relation Projection for Link Prediction: Bridging Message Passing and Triplet Convolution

### Requirements:
python==3.8.5  
torch==1.8.0

### Data:
Download our raw data from Google Drive using the link: https://drive.google.com/file/d/1HYx0wRQgN5MOCmfgY5O_bndiuNroCmuP/view?usp=sharing. Data provided here includes FB15k-237, WN18RR and Kinship.

### Logs
The original training logs can be found in './log/'. The training logs contain information about model training with FB15k-237, WN18RR and Kinship.

### Screenshots
If training logs are too large to be opened, the screenshots of the logs with timestamp are provided under './screenshots/' for you to view our experiments, the datasets, and the final experimental results.

### Training:
When running the model for the first time, please make sure that the param 'get_2hop' is configured as Ture in './config.py'. This will generate 2-hop triples in the data preprocessing stage.
  
After running the model once, please make sure that the param 'get_2hop' is configured as False and 'use_2hop' is configured as True in './config.py'.

Start model training by executing the following command:
$ sh run.sh

### Parameters:
Some of the configurations for the initial parameters are provided in the file './config.py'.  
 
`valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training  

`pretrained_emb`: embeddings initialized from TransE  

`get_2hop`: True only if running for the first time  

`use_2hop`: True if using 2-hop triples for training  

`alpha`: Alpha coefficients for LeakyRelu used in GAT  

`valid_invalid_ratio_conv`: Ratio of valid to invalid triples for ConvKB training  

`margin`: Margin in hingle-loss  

### Acknowledgements
We would like to express our thanks to the authors of the ACL 2019 paper: Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs. We developed our model based on their code:https://github.com/lovishmadaan/kbgat, incorporating relation projection mechanism to bridge relation semantics in massage passing stage and triplet convolution stage.
