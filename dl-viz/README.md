# ESnet Spatiotemporal Graph Deep Learning Model


This is the visual interface of the ESnet SNMP network spatio-temporal traffic prediction model. It is trained using a Recurrent Neural Network architecture model known as Long Short Term Memory (LSTM) and the SNMP traffic data is featurized based on the Autoregressive Integrated Moving Average (ARIMA) time series data analysis paradigm. 

This web app is built using the libraries developed at ESnet including pond.js, react-network-diagrams and react-timeseries-charts that are used in the public facing [ESnet Portal](https://my.es.net).

This web app is still under construction, but to run this production build on a static server, follow these steps

Getting started
---------------
For Mac users only:
Install Homebrew.
```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install Node and NPM
```
brew install node
```

Install Serve
```
npm install -g serve
```

Start web app
```
cd dl-viz
serve
```

Point your browser to:
    http://localhost:3000/

