# seq-to-tree

Libraries required for this model: pytorch, python, numpy, scipy, cuda

Data: #url

Download the data folder from the above location and place it next to src folder in the code repository

Steps to follow:

1. Go to src directtory
2. Execute 'python preprocessing/generate_batches.py
3. Execute 'python train.py'




Using the web interface. In the section Submitting/Task 6a you can find a form with a "Browse" field and a system dropdown menu. After selecting the file in your computer that contains the JSON string and selecting the name of the system that corresponds to these results you can submit them. The format of the JSON string in this case will be:
{"documents": [{"labels":["label1","label2",...,"labelN"], "pmid": 22511223},
                      {"labels":["label1", "label2",...,"labelM"],"pmid":22511224},
                                                .
                                                .
                      {"labels":["label1", "label2",...,"labelK"], "pmid":22511225}]}
where "label1",.."labelN" are the MeSH indicators e.g. "D005260" and not the human annotation i.e. "Female".
