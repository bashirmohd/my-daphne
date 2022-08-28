
--------------------------------------------------------------------------------

Metaroutes is trying to minimize the average flow completion time using MAML-TRPO algorithm. The mission of packet routing is to transfer each packet to its destination through the relaying of multiple routers. The queue of routers follows the first-in first-out (FIFO) criterion. Each router constantly delivers the flow to its neighbor node until that packet reaches its termination.


## Installation

~~~bash
pip install gym
pip install torch
pip install Cython
pip install networkx
pip install cherry-rl
pip install matplotlib
pip install cfg_load[all] --user
~~~

## Run
~~~bash
python metaroutes.py
~~~

## References

~~~
https://github.com/learnables/learn2learn
https://github.com/learnables/cherry
https://github.com/esnet/daphne
~~~





