# NeurIPS Reproducability Challenge: Tensor Monte Carlo

## Requirements
* ```matplotlib```
* ```keras```
* ```tensorflow==1.14```
* ```tensorflow_probability=0.7.0```


## Reproduce results
The main scripts used are ```TMC.py``` and ```IWAE_forward.py```, the results presented in the report were obtained by running these two scripts. The hyper parameter search can be done by configuring ```TMC_hyper_param_search.py``` and using the makefile in a similar manner. In both ```TMC.py``` and ```IWAE_forward.py``` there is the option to restore a model from a previously saved checkpoint and plot reconstructed data. To do this set ```restore_and_recon=True``` in the scripts. To train regularly set it to False.

Examples of how to run these two files:

```python3 TMC.py -k 20 --epochs 400 --batch_size 128 --model_type small```   
```python3 TMC.py -k 20 --epochs 400 --batch_size 128 --model_type large```   

```python3 IWAE_forward.py -k 20 --epochs 400 --batch_size 128 --model_type small```  
```python3 IWAE_forward.py -k 20 --epochs 400 --batch_size 128 --model_type large```   

The hyper parameter search can be done by configuring ```TMC_hyper_param_search.py``` or running the makefile with

```make tmc-hyper-param-srch EPOCHS=400 BATCHSIZE=128 MODELTYPE=small FILE=models/IWAE_forward.py```
