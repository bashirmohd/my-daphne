<p align="center">
<img src="https://github.com/esnet/daphne/blob/master/Network_time_series_library/reports/figures/implementation.png" width="100%" height="100%" title="implementation">
<p>

# NetsLib: A Time-series Prediction libraries for Network traffic and flow bursts prediction using Advance Deep Learning and Statistical approaches.

This folder hosts all the final code for network time-series library for all network operation data, which include data driven learning to predict Wide Area Network (WAN) traffic. We developed a time-series prediction library for Network traffic to predict flow bursts and prevent congestion. We implement a framework to characterize WAN and Internet traffic traces, exploring both statistical and deep learning approaches to produce multi-step predictions with minimal errors, because estimating future traffic can help improve link usage and optimize bandwidth utilization. 


## Instructions 
NetsLib is developed to hosts all the codes for network time-series library for all network operation data. It gives you real-time predictions and shows performance with ARIMA, SARIMA and all deep learning models. To run NetsLib please resolve the following dependencies:


* Tensorflow
* Keras 
* GPy
* apscheduler
* scikit-learn
* PyTorch



## Run
To run NetsLib in online mode use the following command to install all the required dependencies:
```python
command: python Setup.py 
```

## Citing this work

If you use NetsLib for academic or industrial research, please feel free to cite the following [paper1](https://dl.acm.org/doi/abs/10.1145/3391812.3396268), [paper2](https://ieeexplore.ieee.org/abstract/document/8901870):

```

@inproceedings{krishnaswamy2020data,
  title={Data-driven Learning to Predict WAN Network Traffic},
  author={Krishnaswamy, Nandini and Kiran, Mariam and Singh, Kunal and Mohammed, Bashir},
  booktitle={Proceedings of the 3rd International Workshop on Systems and Network Telemetry and Analytics},
  pages={11--18},
  year={2020}
}

@inproceedings{mohammed2019multivariate,
  title={Multivariate Time-Series Prediction for Traffic in Large WAN Topology},
  author={Mohammed, Bashir and Krishnaswamy, Nandini and Kiran, Mariam},
  booktitle={2019 ACM/IEEE Symposium on Architectures for Networking and Communications Systems (ANCS)},
  pages={1--4},
  year={2019},
  organization={IEEE}
}

NetPredict SC19 demo: The International Conference for High Performance Computing, Networking, Storage, and Analysis.
November 17–22, 2019. Network Research Exhibition.

```

## Contacts

* [Mariam Kiran ](https://sites.google.com/lbl.gov/daphne/home?authuser=0)(mkiran@es.net)
* [Nandini Krishnaswamy](https://sites.google.com/lbl.gov/daphne/home?authuser=0)
* [Bashir Mohammed](https://sites.google.com/lbl.gov/daphne/home?authuser=0)


## Acknowledgments

This project is supported by Lawrence Berkeley National Laboratory under DOE Contract number for Deep Learning FP00006145 investigating Large-Scale Deep Learning for Intelligent Networks

