# NeurIPS Reproducibility Challenge: Tensor Monte Carlo

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

## Licence
See LICENCE.txt file.

## DOI
https://doi.org/10.5281/zenodo.3707783

## Citing
@article{linus nilsson_martin larsson_oskar kviman_2020, title={TMC - Tensor Monte Carlo - Reproducibility Report}, DOI={10.5281/zenodo.3707783}, abstractNote={<p>Initial release of the TMC - Tensor Monte Carlo reproducibility repo for the NeurIPS 2019 Reproducibility Challenge.</p>}, publisher={Zenodo}, author={Linus Nilsson and Martin Larsson and Oskar Kviman}, year={2020}, month={Mar}}
