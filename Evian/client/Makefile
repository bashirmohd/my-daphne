CORE_FOLDER=src/core/
NLU_FOLDER=src/nlu

all: train_nlu train_core
	make -C $(CORE_FOLDER) run

test_nlu: 
	make -C $(NLU_FOLDER) test

train_nlu:
	make -C $(NLU_FOLDER) train

train_core:
	make -C $(CORE_FOLDER) train

online: train_nlu
	make -C $(CORE_FOLDER) online

run_app: train_nlu
	make -C $(CORE_FOLDER) run_app

clean:
	make -C $(CORE_FOLDER) clean
	make -C $(NLU_FOLDER) clean

re: clean all
