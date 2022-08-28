Project under development.


# Overview
---------
A simple end-to-end, light-weight tool for testing Network Performance, measuring available throughput and displaying the route across the network along side saving to memory.

# NetPreflight: Testing Network Performance before DTN-DTN transfer (designed for big data transfers)
------------------------------------------------------------------------------

To Add Abstract

* Version 1) Network Measurement Tools used - Iperf, Traceroute.

  ```iperf3 -s ```
  
  ```iperf3 -c <server_ip>```
  
  ```traceroute <server_ip>```

    * Run the following on your DTN terminal:

    * bash <script_name (.sh)> <no._of_runs> <server_ip> <file_transfer_size> <file_output>


  ```e.g bash preflightcheck.sh 5 192.5.87.205 1G testresult.txt```
  

* Version 2) Without Iperf - using password

    * on HOST_A: On your terminal run the command below with the following arguements  

    ```python <script_name> -H <TargetHost_IPaddress>  - F <targetFile> - I <no.of_iterations>```      
                                                                                          
                                                                                        
    ```e.g python preflight_pass.py -H 67.205.158.239 -F /root/largefiles/100MB.zip -I 5```
    
    ```e.g python preflight_pass.py -H 138.68.10.107 -F /root/largefiles/5MB.zip -I 5```
     
    * on HOST_B: No action is required on host_B
   
    * Remember to specify the TargetHost IP address for Example.     

    * user: root  

    * pw: Password1Pass   

    * Requirements: sudo pip install paramiko
     
     
* Version 3) Without Iperf - netpreflight_main - using public keys

    * on HOST_A: On your terminal run the command below with the following arguements.        
                                                                                        
    ```python <scriptname> -H <TargetHostIPaddress> -K <KeyFilepath>  -F <targetFile> -I <no. of iterations> -S <targetFilesize>``` 
                                                                                        
    ```e.g python preflight_keys.py -H 192.5.87.20 -K /home/cc/experiments/uc-mc4n-key.pem -F /home/cc/experiments/5MB.zip -I 5 -S 5```
     
    * on HOST_B: Generate files using dev zero on the Target host(Destination Node). For example to create a 1MB, 1GB:

    ```dd if=/dev/zero of=onemb.zip count=1 bs=1024``` 
    
     ```dd if=/dev/zero of=onegig.zip count=1 bs=10240``` 
     
    * Requirements: sudo pip install paramiko

    
