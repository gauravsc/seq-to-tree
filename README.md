# seq-to-tree 

This is a python implementation of the transformer based seq-to-tree model

Libraries required for this model: pytorch, numpy, scipy, cuda

Data: #url

Download the data folder from the above location and place it next to src folder in the code repository

Steps to follow:

1. Go to src directtory
2. Execute 'python preprocessing/generate_batches.py
3. Execute 'python train.py'



The format of the JSON string to be submitted:

{"documents": [{"labels":["label1","label2",...,"labelN"], "pmid": 22511223},

                      {"labels":["label1", "label2",...,"labelM"],"pmid":22511224},
                                                .
                                                .
                      {"labels":["label1", "label2",...,"labelK"], "pmid":22511225}]}

where "label1",.."labelN" are the MeSH indicators e.g. "D005260" and not the human annotation i.e. "Female".
