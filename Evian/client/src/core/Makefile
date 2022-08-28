.PHONY: init online run

train:
	@/usr/bin/python train_init.py
online:
	@/usr/bin/python train_online.py
run:
	@/usr/bin/python run.py 2> testerr
run_app:
	@/usr/bin/python run_app.py 2> testerr
clean:
	rm -rf models/*
	rm  -rf testerr
