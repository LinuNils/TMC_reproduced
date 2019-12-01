# NeurIPS Reproducability Challenge: Tensor Monte Carlo

## Requirements
* ```matplotlib```
* ```keras```
* ```tensorflow==1.14```
* ```tensorflow_probability=0.7.0```


## Reproduce results
Use the makefile to run different architectures. The main scripts used are ```TMC.py``` and ```IWAE_forward.py```, the results presented in the report were obtained by running these two scripts. The hyper parameter search can be done by configuring ```TMC_hyper_param_search.py```.

```make run K=20 EPOCHS=400 BATCHSIZE=128 FILE=models/IWAE_forward.py PYTHON=python3```

## Practical cheat sheet

### create ssh key
```ssh-keygen -t rsa -b 4096 -C```

Add it to deploy keys for project. 

### Clone and change branch
- ```git clone ...```
- ```git fetch --all```
- ```git checkout ...```

### Run docker
Run docker image:    
- ```sudo docker run -it tensorflow/tensorflow:latest-gpu bash```

Run docker image with gpu:s    
- ```sudo docker run --gpus all -it tensorflow/tensorflow:latest-gpu bash```

Run docker image with gpu:s and mount current folder (```$PWD```) as /temp on docker image.    
```sudo docker run --gpus all -v $PWD:/temp -it tensorflow/tensorflow:latest-gpu bash```
