<p align="center">
<img src="https://github.com/esnet/daphne/blob/master/Chameleon/NetGraf_topology_set_up_on_chameleon/figures/netgraf.png" width="100%" height="100%" title="netgraf">
<p>


# NetGraf: A Collaborative Network Monitoring Automation Stack using Ansible.
This folder contains on all the codes, playbook, inventory, roles and group variables of the NetGraf-ansible automation tool.
Chameleon is a large-scale, deeply reconfigurable experimental platform built to support Computer Sciences systems research. For more details about the manual setup on chameleon without Ansible please visit: [HERE](https://docs.google.com/document/d/1peslsyuAVCAjvZE-aM3DXVG41X8ry9wDytElCcIjJmI/edit)  and to view our paper published recently click:[HERE](https://github.com/esnet/daphne/blob/master/Chameleon/NetGraf_topology_set_up_on_chameleon/NetGraf_SC_2020_Poster_2020_Extended_Abstract_Submission.pdf).


## Architectural workflow on NetGraf using  Ansible:

* Step-1: Define file structure (Best Practice)

* Step-2: Inventory ---- First define your VM details, this will tell Ansible, how to login to VM over SSH.

* Step-3: Group Variable -------

* Step-4: Roles ----  This is the place where you define your automation e.g files, tasks(plays).


## Deployment Steps:

* Create playbook and inventory to run on a local node or VM  on  Chameleon. 
(Inventory: Information regarding VMs/nodes to be managed, Task: Procedure to be executed).
* Create SSH to the target nodes
* Ansible Server gathers the facts of the target nodes on Chameleon to get the indication of the target nodes
* Playbook are sent to the Chameleon nodes.
* Playbook are executed in the Chameleon nodes.
