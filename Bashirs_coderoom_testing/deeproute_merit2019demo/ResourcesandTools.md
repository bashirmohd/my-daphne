# Resources and Steps Required to Setup DeepRoute Controller

We used Jupyter notebook to setup deeproute's experimental infrastructure. 
DeepRoute experiment includes:an OpenFlow controller hosted on Chameleon, an OpenFlow swtich, and two nodes one at CHI@UC(Chicago site) and the other at CHI@TACC(Texas site) both connected to the switch.

* Heat template for  and mininet backend:     
    Included:
* Obtain layer2S circuits between Chi@UC and Chi@TACC to create deeproute network.

* Connect instances to network created above.

* Cross Traffic - Iperf3

```iperf3 -c <server_ip> -t 20 -i 5 -b 900Mbps -u -t 500```

* Controller - RYU OpenFlow
    
    Included: simple_switch_13_custom_chameleon_org.py


### Background: 

DeepRoute on Chameleon is composed of two OpenStack sites with deeply programmable compute, storage, and networking infrastructure. We used the Bring-your-own-controller (BYOC) software defined networking (SDN). 

### Basic Chameleon Networking

Each of the two sites is includes 10 Gbps uplink connectivity to shared core network that can be used to connect Chameleon nodes to core services, other Chameleon nodes, and even external facilities including GENI.  By default, each Chameleon node can access the public Internet through the core network. Advanced experiments can allocate tenant controlled SDN networks and isolated layer2 circuits across the core to connect Chameleon nodes and external facilities. 

### Software Defined Networking on Chameleon

Chameleon’s Bring Your Own Controller (BYOC) functionality enables tenants to create isolated network switches managed using OpenFlow controllers provided by the user. This feature is targeted at users wishing to experiment with software-defined networking (SDN) as well as users with experiments that have non-standard networking requirements. 

#### DeepRoute Environment Setup (Step-by-Step):



* Step 1. Setup DeepRoutes's Controller Environment at CHI@TACC

#Set up user's project (user's can be multiple ones, so there is no default currently)
#Our Project name is MC4N, please replace with your project name on Chameleon.

```export OS_PROJECT_NAME='MC4N'```

#Set region (again, no default currently)

```export OS_REGION_NAME='CHI@TACC'```

#Set chameleon keypair name and path to the private ssh key

```export SSH_KEY_NAME=tac-key```
```export SSH_PRIVATE_KEY=${HOME}/work/bashir-chameleon-jupyter```


#Set the reservations to use.  
#CONTROLLER_RESERVATION_ID can be for any type of node.
#NODE_RESERVATION_ID must be for Skylake nodes.

```export CONTROLLER_RESERVATION_ID='d3ea7ec0-6153-4d3c-bf1f-3badc056bba2'```
```export NODE_RESERVATION_ID='be4980ee-31cd-4c33-8634-ca69fd4fca99'```

*  Step 2. Create DeepRoute's OpenFlow network on Chameleon.

#Set the name of the orcestration stack. We suggest embedding your username or some other identifiable string to make it easier to find you nodes.

```CONTROLLER_STACK_NAME=${USERNAME}"_controller_stack"```

#Set the controller node name. See above about using  identifiable names.

```CONTROLLER_NODE_NAME=${USERNAME}"_controller"```

#Set the network that the controller uses to communicate with the swtich. This networks must be accessible on the Internet and will not be the network that the controller is managing. We suggest using "sharednet1"

```CONTROLLER_NETWORK="sharednet1"```

#Configure the OpenFlow port to be used by the CONTROLLER_OPENFLOW_PORT=6653

```echo Creating controller. This will take several minutes! openstack stack create --max-width 80 \```

```--template "https://www.chameleoncloud.org/appliances/api/appliances/54/template" \```

```--parameter "key_name=${SSH_KEY_NAME}" \``` 

```--parameter "reservation_id=${CONTROLLER_RESERVATION_ID}" \```

```--parameter "ryu_port=${CONTROLLER_OPENFLOW_PORT}" \```

```--parameter "network_name=${CONTROLLER_NETWORK}" \```

```--parameter "controller_name=${CONTROLLER_NODE_NAME}" --wait \```

```${CONTROLLER_STACK_NAME}```
   
```echo Controller creation complete! ```

```echo ${CONTROLLER_NODE_NAME} status `openstack server show  --format value -c status ${CONTROLLER_NODE_NAME}```

```CONTROLLER_PUBLIC_IP=`openstack server show --format value -c addresses ${CONTROLLER_NODE_NAME} | cut -d " " -f 2```

```echo The controller public IP is $CONTROLLER_PUBLIC_IP```

```echo Please wait a few more minutes until the controller is completely configured and ready for logins.```


*  Step 3. View the Controller log file.

```CONTROLLER_PUBLIC_IP=`openstack server show --format value -c addresses ${CONTROLLER_NODE_NAME} | cut -d " " -f 2```

```ssh -i ${SSH_PRIVATE_KEY} \```

```-o UserKnownHostsFile=/dev/null \```

```-o StrictHostKeyChecking=no \```

```cc@${CONTROLLER_PUBLIC_IP} \```

```tail -n 20 /var/log/ryu/ryu-manager.log ```

*  Step 4. Start the Openflow switches.

#Set the subnet to use on the OpenFlow network

```OPENFLOW_NETWORK_SUBNET_CIDR="192.168.100.0/24"```

#Set the OpenStack names for the network, subnet, router, and switch. 
#See above about using identifiable names.  
#Note that OPENFLOW_SWITCH_NAME cannot include the '-' character.

```OPENFLOW_NETWORK_NAME=${USERNAME}"Network"```


```OPENFLOW_SUBNET_NAME=${USERNAME}"Subnet"```

```OPENFLOW_ROUTER_NAME=${USERNAME}"Router"```

#Note that OPENFLOW_SWITCH_NAME cannot include the '-' character. 
#This name is used to add additional uplink ports to the same OpenFlow switch.

```OPENFLOW_SWITCH_NAME=${USERNAME}"Switch"```

```echo Creating network ${OPENFLOW_NETWORK_NAME}```

```openstack network create --max-width 80 \```

```--provider-network-type vlan \```

```--provider-physical-network physnet1 \```

```--descriptionofController=${CONTROLLER_PUBLIC_IP}:${CONTROLLER_OPENFLOW_PORT},VSwitchName=${OPENFLOW_SWITCH_NAME}\```

```${OPENFLOW_NETWORK_NAME}```
                         
```PRIMARY_UPLINK_VLAN=`openstack network show -c provider:segmentation_id -f value ${OPENFLOW_NETWORK_NAME}```

```echo Primary uplink VLAN and port ID: $PRIMARY_UPLINK_VLAN ``` 


*  Step 5. Add a subnet and Router to the DeepRoute Network

```echo Creating Subnet ```

```openstack subnet create --max-width 80 \```

```--subnet-range ${OPENFLOW_NETWORK_SUBNET_CIDR}--dhcp \```

```${OPENFLOW_SUBNET_NAME}```
                        
```echo Creating Router```

```openstack router create --max-width 80 ${OPENFLOW_ROUTER_NAME}```

```echo Linking router to subnet```

```openstack router add subnet ${OPENFLOW_ROUTER_NAME} ${OPENFLOW_SUBNET_NAME}```

```echo Linking router to external gateway```

```openstack router set --external-gateway public ${OPENFLOW_ROUTER_NAME}```

```echo Network ${OPENFLOW_NETWORK_NAME} is ready for nodes!```



*  Step 6.Launch servers connected to the new created network.

```echo Creating servers... This will take several minutes!```

```openstack server create --max-width 80 \```

```--flavor "baremetal" \```

```--image "CC-CentOS7" \```

```--key-name "tac-key" \```

```--hint reservation=${NODE_RESERVATION_ID} \```

```--security-group default \```

```--nic net-id="${OPENFLOW_NETWORK_NAME}" \```

```--min 2 \```

```--max 2 \```

```--wait \```

```${USERNAME}-node```

```echo Server creation complete!```

```echo ${USERNAME}-node-4 is `openstack server show --format value -c status ${USERNAME}-node-1```

```echo ${USERNAME}-node-5 is `openstack server show --format value -c status ${USERNAME}-node-2```


* NB: remember to clean up your resources. 



