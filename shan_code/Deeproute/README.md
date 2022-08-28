# Deeproute Gym environment



## Installation

~~~bash
pip install gym
pip install Cython
pip install networkx
pip install matplotlib
pip install cfg_load[all] --user
~~~

## Run
Actions are selected using the shortest path algorithm (Dijkastra).
~~~bash
python deeproute_spa_agent.py
~~~


Actions are randomly selected:
~~~bash
python deeproute_random_agent.py
~~~

