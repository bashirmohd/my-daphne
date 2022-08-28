<p align="center">
<img src="https://github.com/esnet/daphne/blob/master/Documentation_scripts_master/dtn_network_measurement/figures/cham_topo.png" width="100%" height="100%" title="cham_topo">
<p>

# DTN Large Data Flow Transfer Network Measurement Automation Experiment
This is a DTN Large Data Flows transfer Network Measurement Experiment between the University of Chicago(UC) and Texas Advanced Computing Center(TACC) Chameleon Site. The aim of the experiment is automate Large Data Flows transfer network measurement.


# Resources for Reproducibilty
* We prepared a data dransfer node (DTN) that can be used to provide efficient network data transfer over a long fat network at both the University of Chicago(UC) and Texas Advanced Computing Center(TACC) Chameleon Site. Each DTN comes with an Ubuntu 16.04 node with network stack optimization for 10 Gbps.

     
     Included: dtnscript.bash
    * Create a DTN Client/Tester Compute Node at UC
    * Create a DTN Server Compute Node at TACC
    * Connect instances to shared network created.
    
* Bare Metal Nodes:
    * One DTN Server (Compute Node-CHI@UC)
    * One DTN Client/Tester(Compute Node-CHI@TACC)
    * All Compute Nodes - CC-Ubuntu 16.04
    
* Network
    * Two Corsa switches: TACC and UC
    * Network stack optimization link between (CHI@TACC) is 10 Gbps

* Network Measurement Tools - Iperf, Traceroute.

    ```iperf3 -c <server_ip>```

    ```traceroute <server_ip>```

# To Test the DTN script: 
     

* Login to your DTN Sever node @CHI@UC site:

  ```ssh -i ~/.ssh/uc-mc4n-key.pem cc@192.5.87.205```


* Run the following on your UC node terminal: 

   ```iperf -s -D```


* Login to your DTN Client/Tester node @CHI@TACC site:

  ```ssh -i ~/.ssh/tacc-mc4n-key.pem cc@129.114.109.242```

  Copy the ```dtnscript.sh``` file into your node @CHI@TACC site or create a new file, and paste the content of script into it:

   ```vim dtnscript.sh```
 
* Run the following on your TACC node terminal:

   ```bash <script_name> <no._of_runs> <server_ip> <file_transfer_size> <file_output>```

   ```e.g bash dtnscript.sh 5 192.5.87.205 1G file.txt```

   Please note that your results and your log file will be saved in the same directory where you have your bash script.

* Steps: 

        Repeat the same steps above but using different file transfer sizes e.g 0.1G, 0.2G, 1GB, 2GB, 3GB, etc


# To Test netpreflight_ssh.py script using ssh-access between two servers.(Without Iperf): 

* Destination IP address: 192.5.87.127
* Source IP address: 192.5.87.167


* Login to your Destination Sever node @CHI@UC site:

  ```ssh -i ~/.ssh/uc-mc4n-key.pem cc@192.5.87.127```

* Upload an image file and take note of the directory path: 

* To check if ssh port 22 is open and listening, use the following command : 

   ``` netstat -tuplen```


* Login to your Source Server node @CHI@UC site:

  ```ssh -i ~/.ssh/uc-mc4n-key.pem cc@192.5.87.167```

  Copy the ```netpreflight_ssh.py``` file into your Source Server node or create a new file, and paste the content of script into it:

   ```vim netpreflight_ssh.py```
 
* Run the following on your Source Server node terminal. Specify the directory(Optional):

   ```python <scriptname e.g netpreflight_ssh.py> -H <targetHost> -F <targetFile> -I <iterations>```

   ```e.g python netpreflight_ssh.py -H 192.5.87.127 -F d-icon.png -I 3```

   Please note that your results and your log file will be saved in the same directory where you ran your script.

* Run the following to view your results and downloaded file: 

  ```ls```

  ```cat outfile```


## Contacts

* [Mariam Kiran ](https://sites.google.com/lbl.gov/daphne/home?authuser=0)
* [Bashir Mohammed](https://sites.google.com/lbl.gov/daphne/home?authuser=0)
        
   
