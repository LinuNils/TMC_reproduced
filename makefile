K?=20
EPOCHS?=1200
BATCHSIZE?=128
FILE?=models/IWAE_forward.py
MODELTYPE?=small
PYTHON?=python3


tmc-hyper-param-srch:
	for k in 1 5 ; do \
		for lr in 0.00001 0.0001 0.001 0.01 ; do \
				echo "##################################################"  ; \
				echo "Running $(FILE) with the following settings"  ; \
				echo K = $$k ; \
				echo EPOCHS = $(EPOCHS) ; \
				echo BATCHSIZE = $(BATCHSIZE) ; \
				echo FILE = $(FILE) ; \
				echo MODELTYPE = $(MODELTYPE) ; \
				echo LEARNING_RATE = $$lr ; \
				echo "##################################################"  ; \
				sleep 2 ; \
				$(PYTHON) $(FILE) --model_type $(MODELTYPE) -k $$k --batch_size $(BATCHSIZE) --epochs $(EPOCHS) --learning_rate $$lr  ; \
				sleep 30 ; \
		done ; \
	done
