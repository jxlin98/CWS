# CWS

This is a CWS implementation using bert for pku dataset


## Requirements
* Python (tested on 3.7.11)
* numpy (tested on 1.21.5)
* CUDA (tested on 11.3)
* PyTorch (tested on 1.10.1)
* Transformers (tested on 4.15.0)

## Files
* bert-base-chinese is the bert pretrain model for chinese
* checkpoint saves the best model parameter
* datasets contains the pku CWS dataset
* result saves the test result and score
* scripts contains the script for evaluation

* data.py reads the pku CWS dataset
* model.py implements the model
* train.py and test.py are for train and test
* utils.py contains some data structure and functions
* result.sh runs the script for evaluation

## Train and Test

```bash
>> python train.py  # for train
>> python test.py  # for test and the result will be saved in result dir
>> bash result.sh  # run the script for evaluation
>> cat result/score.txt  # see the result
```