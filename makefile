K?=20
EPOCHS?=1200
BATCHSIZE?=128
FILE?=models/IWAE_forward.py
MODELTYPE?=small
PYTHON?=python3

single:
	echo "##################################################"  ; \
	echo "Running $(FILE) with the following settings"  ; \
	echo K = $(K) ; \
	echo EPOCHS = $(EPOCHS) ; \
	echo BATCHSIZE = $(BATCHSIZE) ; \
	echo FILE = $(FILE) ; \
	echo "##################################################"  ; \
	$(PYTHON) $(FILE) -k $(K) --batch_size $(BATCHSIZE) --epochs $(EPOCHS)  ; \


run:
	for k in 5 20 50 100 ; do \
		echo "##################################################"  ; \
		echo "Running $(FILE) with the following settings"  ; \
		echo K = $$k ; \
		echo EPOCHS = $(EPOCHS) ; \
		echo BATCHSIZE = $(BATCHSIZE) ; \
		echo FILE = $(FILE) ; \
		echo "##################################################"  ; \
		sleep 2 ; \
		$(PYTHON) $(FILE) -k $$k --batch_size $(BATCHSIZE) --epochs $(EPOCHS)  ; \
		sleep 30 ; \
	done


tmc-single:
		echo "##################################################"  ; \
		echo "Running $(FILE) with the following settings"  ; \
		echo K = $(K) ; \
		echo EPOCHS = $(EPOCHS) ; \
		echo BATCHSIZE = $(BATCHSIZE) ; \
		echo FILE = $(FILE) ; \
		echo MODELTYPE = $(MODELTYPE) ; \
		echo "##################################################"  ; \
		$(PYTHON) $(FILE) -k $(K) --model_type $(MODELTYPE) --batch_size $(BATCHSIZE) --epochs $(EPOCHS)  ; \



tmc-run:
		for k in 5 20 50 100 ; do \
				echo "##################################################"  ; \
				echo "Running $(FILE) with the following settings"  ; \
				echo K = $$k ; \
				echo EPOCHS = $(EPOCHS) ; \
				echo BATCHSIZE = $(BATCHSIZE) ; \
				echo FILE = $(FILE) ; \
				echo MODELTYPE = $(MODELTYPE) ; \
				echo "##################################################"  ; \
				sleep 2 ; \
				$(PYTHON) $(FILE) --model_type $(MODELTYPE) -k $$k --batch_size $(BATCHSIZE) --epochs $(EPOCHS)  ; \
				sleep 30 ; \
		done


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
