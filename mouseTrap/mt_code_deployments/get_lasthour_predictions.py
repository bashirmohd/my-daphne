
import time
import json
import numpy as np
from datetime import datetime
from google.cloud import storage
from google.cloud import firestore
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import math

def inverse_transform(pred, scaler):    
    return scaler.inverse_transform(pred)


def getlastHourPredictionsData():
    
    start_time=time.time()

    dashfile=open('netpredictdashboard.html','w')

    message= """<html>
  <head>


    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">

      // Load Charts and the corechart and barchart packages.
      google.charts.load('current', {'packages':['corechart']});

      // Draw the pie chart and bar chart when Charts is loaded.
      google.charts.setOnLoadCallback(drawChart);
      
      function drawChart() {

      
      
      """
    
    # one link
    msechickansstr="""var ckmse=new google.visualization.DataTable();
        ckmse.addColumn('string', 'TestRows');
        ckmse.addColumn('number', 'MSE');
        ckmse.addRows(["""
   

    linkchickans=pd.DataFrame(columns=['Time','Actual','Predicted'])
    chickansstr="var chickansdata = new google.visualization.arrayToDataTable([ ['Time', 'Actual', 'Predicted'],"
 

    db = firestore.Client()
    users_ref = db.collection(u'5minpredictions').where("src","==","CHIC").where("dest","==","KANS")
    totalcalnum=0
    count1=0

    for doc in users_ref.stream():
        #print(u'{} => {}'.format(doc.id, doc.to_dict()))

        receivedDict=doc.to_dict()

        propTime=str(time.ctime(receivedDict['lasttimestamp']))
        chickansstr=chickansstr+"['"+propTime+"',"

        linkchickans=linkchickans.append({'Time': propTime, 'Actual': receivedDict['lasttraffic'], 'Predicted': receivedDict['predictedtraffic']},ignore_index=True)
        chickansstr=chickansstr+str(receivedDict['lasttraffic'])+","
        chickansstr=chickansstr+str(receivedDict['predictedtraffic'])+"],"
        
    

        calnumdiff=receivedDict['lasttraffic']-receivedDict['predictedtraffic']
        calnum=math.pow(calnumdiff,2)
        totalcalnum=totalcalnum+calnum
        count1=count1+1
        msechickansstr=msechickansstr+"['"+propTime+"',"
        msechickansstr=msechickansstr+str(calnum)+"],"


        print(receivedDict['timestamp'],receivedDict['predictedtraffic'], receivedDict['lasttimestamp'],receivedDict['lasttraffic']
        , sep=',',end='\n')
        
        #print(datetime.fromtimestamp(receivedDict['timestamp']).isoformat())
    
     

    #create MSE graph


    chickansstr=chickansstr+ "]);"



    msechickansstr=msechickansstr+ "]);"

    chickansstr=chickansstr+msechickansstr
    chickansstr=chickansstr+"""
    var barchart_optionsck = {title:'MeanSquareError:"""+str(totalcalnum/count1)+"""',
                       width:400,
                       height:300,
                       legend: 'none'};
        var barchartck = new google.visualization.BarChart(document.getElementById('barchart_divck'));
		barchartck.draw(ckmse, barchart_optionsck);
    """
    
    chickansstr=chickansstr+"var line_optionsck = {title:'Chic->Kans',width: 900,   height: 500,		legend: { position: 'bottom' }};var linechartck = new google.visualization.LineChart(document.getElementById('curve_chartck'));linechartck.draw(chickansdata, line_optionsck);"

# two link
    msedenvsacrstr="""var dsmse=new google.visualization.DataTable();
        dsmse.addColumn('string', 'TestRows');
        dsmse.addColumn('number', 'MSE');
        dsmse.addRows(["""
   

    linkdenvsacr=pd.DataFrame(columns=['Time','Actual','Predicted'])
    denvsacrstr="var denvsacrdata = new google.visualization.arrayToDataTable([ ['Time', 'Actual', 'Predicted'],"
 

    db = firestore.Client()
    users_ref = db.collection(u'5minpredictions').where("src","==","DENV").where("dest","==","SACR")
    totalcalnum2=0
    count2=0

    for doc in users_ref.stream():
        #print(u'{} => {}'.format(doc.id, doc.to_dict()))

        receivedDict=doc.to_dict()

        propTime=str(time.ctime(receivedDict['lasttimestamp']))
        denvsacrstr=denvsacrstr+"['"+propTime+"',"

        linkdenvsacr=linkdenvsacr.append({'Time': propTime, 'Actual': receivedDict['lasttraffic'], 'Predicted': receivedDict['predictedtraffic']},ignore_index=True)
        denvsacrstr=denvsacrstr+str(receivedDict['lasttraffic'])+","
        denvsacrstr=denvsacrstr+str(receivedDict['predictedtraffic'])+"],"
        
    

        calnumdiff=receivedDict['lasttraffic']-receivedDict['predictedtraffic']
        calnum=math.pow(calnumdiff,2)
        totalcalnum2=totalcalnum2+calnum
        count2=count2+1
        msedenvsacrstr=msedenvsacrstr+"['"+propTime+"',"
        msedenvsacrstr=msedenvsacrstr+str(calnum)+"],"


        print(receivedDict['timestamp'],receivedDict['predictedtraffic'], receivedDict['lasttimestamp'],receivedDict['lasttraffic']
        , sep=',',end='\n')
        
        #print(datetime.fromtimestamp(receivedDict['timestamp']).isoformat())
    
        #create MSE graph


    denvsacrstr=denvsacrstr+ "]);"



    msedenvsacrstr=msedenvsacrstr+ "]);"

    denvsacrstr=denvsacrstr+msedenvsacrstr
    denvsacrstr=denvsacrstr+"""
    var barchart_optionsds = {title:'MeanSquareError:"""+str(totalcalnum2/count2)+"""',
                       width:400,
                       height:300,
                       legend: 'none'};
        var barchartds = new google.visualization.BarChart(document.getElementById('barchart_divds'));
		barchartds.draw(dsmse, barchart_optionsds);
    """
    
    denvsacrstr=denvsacrstr+"""
    var line_optionsds = {title:'Denv->Sacr',width: 900,   height: 500,		    
    legend: { position: 'bottom' }};
    
    var linechartds = new google.visualization.LineChart(document.getElementById('curve_chartds'));    
    linechartds.draw(denvsacrdata, line_optionsds);""" 


# three link
    mseboisdenvstr="""var bdmse=new google.visualization.DataTable();
        bdmse.addColumn('string', 'TestRows');
        bdmse.addColumn('number', 'MSE');
        bdmse.addRows(["""
   

    linkboisdenv=pd.DataFrame(columns=['Time','Actual','Predicted'])
    boisdenvstr="var boisdenvdata = new google.visualization.arrayToDataTable([ ['Time', 'Actual', 'Predicted'],"
 

    db = firestore.Client()
    users_ref = db.collection(u'5minpredictions').where("src","==","BOIS").where("dest","==","DENV")
    totalcalnum3=0
    count3=0

    for doc in users_ref.stream():
        #print(u'{} => {}'.format(doc.id, doc.to_dict()))

        receivedDict=doc.to_dict()

        propTime=str(time.ctime(receivedDict['lasttimestamp']))
        boisdenvstr=boisdenvstr+"['"+propTime+"',"

        linkboisdenv=linkboisdenv.append({'Time': propTime, 'Actual': receivedDict['lasttraffic'], 'Predicted': receivedDict['predictedtraffic']},ignore_index=True)
        boisdenvstr=boisdenvstr+str(receivedDict['lasttraffic'])+","
        boisdenvstr=boisdenvstr+str(receivedDict['predictedtraffic'])+"],"
        
    

        calnumdiff=receivedDict['lasttraffic']-receivedDict['predictedtraffic']
        calnum=math.pow(calnumdiff,2)
        totalcalnum3=totalcalnum3+calnum
        count3=count3+1
        mseboisdenvstr=mseboisdenvstr+"['"+propTime+"',"
        mseboisdenvstr=mseboisdenvstr+str(calnum)+"],"


        print(receivedDict['timestamp'],receivedDict['predictedtraffic'], receivedDict['lasttimestamp'],receivedDict['lasttraffic']
        , sep=',',end='\n')
        
        #print(datetime.fromtimestamp(receivedDict['timestamp']).isoformat())
    
        #create MSE graph


    boisdenvstr=boisdenvstr+ "]);"



    mseboisdenvstr=mseboisdenvstr+ "]);"

    boisdenvstr=boisdenvstr+mseboisdenvstr
    boisdenvstr=boisdenvstr+"""
    var barchart_optionsbd = {title:'MeanSquareError:"""+str(totalcalnum3/count3)+"""',
                       width:400,
                       height:300,
                       legend: 'none'};
        var barchartbd = new google.visualization.BarChart(document.getElementById('barchart_divbd'));
		barchartbd.draw(bdmse, barchart_optionsbd);
    """
    
    boisdenvstr=boisdenvstr+"""
    var line_optionsbd = {title:'Denv->Kans (Model 1)',width: 900,   height: 500,		    
    legend: { position: 'bottom' }};
    
    var linechartbd = new google.visualization.LineChart(document.getElementById('curve_chartbd'));    
    linechartbd.draw(boisdenvdata, line_optionsbd);""" 

    #4th link
    # 
    msedenvkansstr="""var dkmse=new google.visualization.DataTable();
        dkmse.addColumn('string', 'TestRows');
        dkmse.addColumn('number', 'MSE');
        dkmse.addRows(["""
   

    linkdenvkans=pd.DataFrame(columns=['Time','Actual','Predicted'])
    denvkansstr="var denvkansdata = new google.visualization.arrayToDataTable([ ['Time', 'Actual', 'Predicted'],"
 

    db = firestore.Client()
    users_ref = db.collection(u'5minpredictions').where("src","==","DENV").where("dest","==","KANS")
    totalcalnum4=0
    count4=0

    for doc in users_ref.stream():
        #print(u'{} => {}'.format(doc.id, doc.to_dict()))

        receivedDict=doc.to_dict()

        propTime=str(time.ctime(receivedDict['lasttimestamp']))
        denvkansstr=denvkansstr+"['"+propTime+"',"

        linkdenvkans=linkdenvkans.append({'Time': propTime, 'Actual': receivedDict['lasttraffic'], 'Predicted': receivedDict['predictedtraffic']},ignore_index=True)
        denvkansstr=denvkansstr+str(receivedDict['lasttraffic'])+","
        denvkansstr=denvkansstr+str(receivedDict['predictedtraffic'])+"],"
        
    

        calnumdiff=receivedDict['lasttraffic']-receivedDict['predictedtraffic']
        calnum=math.pow(calnumdiff,2)
        totalcalnum4=totalcalnum4+calnum
        count4=count4+1
        msedenvkansstr=msedenvkansstr+"['"+propTime+"',"
        msedenvkansstr=msedenvkansstr+str(calnum)+"],"


        print(receivedDict['timestamp'],receivedDict['predictedtraffic'], receivedDict['lasttimestamp'],receivedDict['lasttraffic']
        , sep=',',end='\n')
        
        #print(datetime.fromtimestamp(receivedDict['timestamp']).isoformat())
    
        #create MSE graph


    denvkansstr=denvkansstr+ "]);"



    msedenvkansstr=msedenvkansstr+ "]);"

    denvkansstr=denvkansstr+msedenvkansstr
    denvkansstr=denvkansstr+"""
    var barchart_optionsdk = {title:'MeanSquareError:"""+str(totalcalnum4/count4)+"""',
                       width:400,
                       height:300,
                       legend: 'none'};
        var barchartdk = new google.visualization.BarChart(document.getElementById('barchart_divdk'));
		barchartdk.draw(dkmse, barchart_optionsdk);
    """
    
    denvkansstr=denvkansstr+"""
    var line_optionsdk = {title:'Denv->Kans (Model 2)',width: 900,   height: 500,		    
    legend: { position: 'bottom' }};
    
    var linechartdk = new google.visualization.LineChart(document.getElementById('curve_chartdk'));    
    linechartdk.draw(denvkansdata, line_optionsdk);""" 

    #total all links

    chickansstr=chickansstr+denvsacrstr
    chickansstr=chickansstr+boisdenvstr
    chickansstr=chickansstr+denvkansstr

    chickansstr=chickansstr+"""
    	
      }
</script>
<body>

	
	<h1>NetPredict Trust Dashboard: Different Deep Learning Models</h1>

    <table class="columns">
      <tr>
         <td> <div id="curve_chartck" style="width: 900px; height: 500px"></div></td>
   		<td><div id="barchart_divck" style="border: 1px solid #ccc"></div></td>

	 </tr>

      <tr>
         <td> <div id="curve_chartds" style="width: 900px; height: 500px"></div></td>
   		<td><div id="barchart_divds" style="border: 1px solid #ccc"></div></td>

	 </tr>
 	</table>


	<h1>Swapping Deep Learning Models</h1>
<table class="columns">

      <tr>
         <td> <div id="curve_chartbd" style="width: 900px; height: 500px"></div></td>
   		<td><div id="barchart_divbd" style="border: 1px solid #ccc"></div></td>

	 </tr>
        <tr>
         <td> <div id="curve_chartdk" style="width: 900px; height: 500px"></div></td>
   		<td><div id="barchart_divdk" style="border: 1px solid #ccc"></div></td>

	 </tr>
	 <tr>
		<td> 
			<div id="curve_chart" style="width: 900px; height: 500px"></div>
			<div id="curve_chart2" style="width: 900px; height: 500px"></div>

		</td>

	  </tr>

	</table>
	<h2>ESnet's Correlation Matrix</h2>
	<img src="1hrfulldata_corr_plot.png" alt="Traffic Correlation Matrix">

  </body>
</html>
    """

    message=message+chickansstr
   
    end_time=time.time()

    dashfile.write(message)
    dashfile.close()
    total_time=end_time-start_time
    print("total_time: %s seconds" %total_time)

#create local json /csv file
getlastHourPredictionsData()