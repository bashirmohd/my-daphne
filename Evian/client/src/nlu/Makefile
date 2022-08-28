all: train test

train:
	/usr/bin/python nlu_train.py

test:
	/usr/bin/python nlu_test.py

clean:
	rm -rf projects

re: clean all

.PHONY: test
