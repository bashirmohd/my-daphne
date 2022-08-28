# Evian : Enabling Various Intents for Autonomic Networks#

The Advanced Light Source is a Department of Energy-funded synchrotron facility that provides access to the brightest beams of soft X-rays. For an experiment,  a physicist will use those X-rays and will send it to a diamond. Then it will create raster images from this experiment. To keep these images for a future utilization, the physicist needs to store them in a database. In order to do that, the physicist calls a network engineer to set up the network for the transfer. We simply this process. We compared intent-based networking projects (INDIRA, OpenDayLight NIC, ONOS).  We found that no existing project was using Natural language processing to get a high-level language intent from the user.  The final goal is to design an autonomic system that will understand the user’s intent, get all the requirements and will be aware about the current and future states of the network to do an intelligent installation of the user's intent.

# Folders #
## client ##
Client side based on RASA, an open-source natural language processing project
## server ##
Server side based on Flask, a micro-framework
## results ##
Some results about Evian

# Installation #

## Packages to install ##
* spaCy
* rasa_nlu
* rasa_core version 10.0
* flask
* flask-restful
* tensorflow
* graphviz 

## installation ##
* Install the packages
* Install the english language package for spaCy : `python -m spacy download en`
* Test if the natural language understanding module works : 
	* Go in the folder *client* and train the nlu model : `make train_nlu`
	* In the same folder test the nlu model : `make test_nlu`
* Train and test the bot locally
	* Go in the folder *client* and train the core model `make train_core`
	* Test how the model works locally `make run`
	* Test how the model works online go in the folder `make online`
* Connection with Slack
	* Give the address of the bot to the slack api
	* Change credentials in the file `src/core/run_app.py`
* Run the bot on Slack
	* Go in the client folder and do `make run_app`

# Future work #
* Connection with Slack/Control of the intent by the Network Engineer thank to Slack
* Linking Evian with Sense
* Secure Connection to Sense (AuthN and AuthZ)
* **Think about all the security issues to reassure network engineers and the security team**. 
	Critical for the deployment of Evian
* Group/User Account -- Ressources Management
* Intent State Machine to handle outage events
* Listening from the network - Many questions about this (How to listen, what to listen, and what to do with the extracted information ?)
* Policy and Conflict Checking
* Develop more complex intents
* Design RDF3
* Evian 2
	* Deep Learning Network Predictions
	* For some tools, deep learning optimizations
	* Develop Evian for many others rendering tools
