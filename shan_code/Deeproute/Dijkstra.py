import sys

def maxfringe(bw,Q):
	target = Q[0]
	temp_bw = 0
	for node_name in Q:
	    if bw[node_name] > temp_bw:
	        target = node_name
	        temp_bw = bw[node_name]
	    
	return target

def dijkstra(nodes, links, nodes_connected_links, links_avail, s, t):
	bw = {}
	Q = []
	prev = {}
	for node in nodes:
	    bw[node.name] = 0
	    prev[node.name] = None
	    Q.append(node.name)

	bw[s] = sys.maxsize
	
	while len(Q) > 0:
	    u = maxfringe(bw, Q)
	    Q.remove(u)
	    for to_link, to_node_name in nodes_connected_links[u]:
	        if to_node_name in Q:
	            if min(links_avail[to_link.name], bw[u]) > bw[to_node_name]:
	                bw[to_node_name] = min(links_avail[to_link.name], bw[u])
	                prev[to_node_name] = u
	
	while prev[t] != s:
	    t = prev[t]
	    
	action = 0
	for to_link, to_node_name in nodes_connected_links[s]:
	    if to_node_name == t:
	        return action
	    else:
	        action += 1

	return False
	
def minfringe(lat,Q):
	target = Q[0]
	temp_lat = sys.maxsize
	for node_name in Q:
	    if lat[node_name] < temp_lat:
	        target = node_name
	        temp_lat = lat[node_name]
	    
	return target
	
def dijkstra_lat(nodes, links, nodes_connected_links, s, t):
	lat = {}
	Q = []
	prev = {}
	for node in nodes:
	    lat[node.name] = sys.maxsize
	    prev[node.name] = None
	    Q.append(node.name)

	lat[s] = 0
	
	while len(Q) > 0:
	    u = minfringe(lat, Q)
	    Q.remove(u)
	    for to_link, to_node_name in nodes_connected_links[u]:
	        if to_node_name in Q:
	            if (to_link.lat + lat[u]) < lat[to_node_name]:
	                lat[to_node_name] = to_link.lat + lat[u]
	                prev[to_node_name] = u
	
	while prev[t] != s:
	    t = prev[t]
	    
	action = 0
	for to_link, to_node_name in nodes_connected_links[s]:
	    if to_node_name == t:
	        return action
	    else:
	        action += 1

	return False
	
