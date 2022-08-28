#main.py
#!/usr/bin/python3

import sys
import os
from PyQt5 import QtCore, QtWidgets, QtGui 
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtWidgets import QPushButton, QFrame, QAction
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QGroupBox

from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtCore import QSize    
from PyQt5.QtCore import QObject, Qt, pyqtSignal

from graphviz import Digraph
import graphviz as gv
import pylab
import pydot

#from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import collections
import random
import time
import math
import numpy as np


from sklearn.datasets import load_iris
from sklearn import tree



import copy
import tensorflow as tf

#from examples.tetris import Env
from collections import deque
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential

#from methodlibrary.dqn import DQNAgent


class MainWindow(QMainWindow):
	
	def __init__(self):

		QMainWindow.__init__(self)

		#self.centralWidget=QtWidgets.QWidget(self)
		
		#all buttons initialized
		buttontopo = QPushButton("Load Topology")
		buttonbuild = QPushButton("Load network metrics")
		buttonwrite = QPushButton("Write graph")

		buttondata = QPushButton("Load URL for data")
		buttondatagen = QPushButton("Load data")

		buttontime = QPushButton("Classify - traffic")
		buttonlook = QPushButton("Look Inside")

		buttonTrain= QPushButton("Train Box")
		buttonOutput = QPushButton("Performance improvements")
		
		labeltopo=QLabel("Topology")
		labeldata=QLabel("Data")
		labelana=QLabel("Analysis")
		labelsim=QLabel("Simulate")


		#All groups boxes addes

		#groupbox1: Topology
		topoBox=QVBoxLayout()
		
		topoBox.addWidget(labeltopo)
		topoBox.addWidget(buttontopo)
		topoBox.addWidget(buttonbuild)
		topoBox.addWidget(buttonwrite)
		#topoBox.setColor(Qt.blue)

		mainMenu = self.menuBar()
		fileMenu = mainMenu.addMenu('Examples')

		listExamplesAction = QAction("Classify", self)
		listExamplesAction.triggered.connect(self.menuclicked)

		listExamplesAction2 = QAction("Time Prediction", self)
		listExamplesAction2.triggered.connect(self.menuclicked)


		listExamplesAction3 = QAction("Reinforcement Learning (DQN)", self)
		listExamplesAction3.triggered.connect(self.menuclicked)
		#extractAction.setShortcut("Ctrl+Q")
		#extractAction.setStatusTip('Leave The App')
		#extractAction.triggered.connect(self.close_application)

		fileMenu.addAction(listExamplesAction)
		fileMenu.addAction(listExamplesAction2)
		fileMenu.addAction(listExamplesAction3)
		self.dialogs=list()

		#self.statusBar()
		buttontopo.clicked[bool].connect(self.buttonclicked)

		#groupbox2: Data
		dataBox=QVBoxLayout()

		dataBox.addWidget(labeldata)
		dataBox.addWidget(buttondata)
		dataBox.addWidget(buttondatagen)

		#groupbox3: Analysis
		anaBox=QVBoxLayout()

		anaBox.addWidget(labelana)
		anaBox.addWidget(buttontime)
		anaBox.addWidget(buttonlook)

		#groupbox4: simulate
		simBox=QVBoxLayout()

		simBox.addWidget(labelsim)
		simBox.addWidget(buttonTrain)
		simBox.addWidget(buttonOutput)

	
		#self layout
		mainLayout=QVBoxLayout()

		mainLayout.addLayout(topoBox)
		mainLayout.addLayout(dataBox)
		mainLayout.addLayout(anaBox)
		mainLayout.addLayout(simBox)

		parentWidget=QWidget()
		parentWidget.setLayout(mainLayout)
		parentWidget.setFixedSize(QSize(250,450))

		self.setCentralWidget(parentWidget)

		self.setGeometry(0, 0, 500, 300)
		self.setWindowTitle("Hecate")
		self.show()


		
	def createGridLayout(self):
		self.verticalGroupBox=QGroupBox("Services")
		layout=QGridLayout()
		layout.setColumnStretch(1,4)


		vBoxLayout=QVBoxLayout()
		button1 = QPushButton("Football", self)
		button1.clicked.connect(self.buttonclicked)
		vBoxLayout.addWidget(button1)
		

		v2BoxLayout=QVBoxLayout()
		button12 = QPushButton("F22ootball", self)
		button12.clicked.connect(self.buttonclicked)
		v2BoxLayout.addWidget(button12)
		
		self.groupBox.setLayout(vBoxLayout)
		self.groupBox.setLayout(v2BoxLayout)


		
		

	def buttonclicked(self,pressed):
		source=self.sender()

		if pressed:
			val=2
		else: 
			val =0

		if source.text()=="Load Topology":
			print('Clicked topo button.')
		elif source.text()=="Attach Data":
			print('Clicked data')
		else:
			print('Clicked block')

	def menuclicked(self,pressed):
		source=self.sender()
		if source.text()=="Classify":
			print('draw')			

			#draw dot graph
			dotclassify=gv.Graph(format='png')
			dotclassify.attr(rankdir='LR', size='10')
			dotclassify.node('iris.data')
			dotclassify.node('DecisionTree Classifier')
			dotclassify.node('Optimize parameters')
			dotclassify.node('Output Error/Model', shape ='double circle')

			dotclassify.edge('iris.data', 'DecisionTree Classifier')
			dotclassify.edge('DecisionTree Classifier', 'Optimize parameters')
			dotclassify.edge('Optimize parameters', 'DecisionTree Classifier')
			dotclassify.edge('Optimize parameters', 'Output Error/Model')
			#dotclassify.view()
			#print(dotclassify.source) 
			filename=dotclassify.render(filename='dotclassify.dot')
			#pylab.savefig('filename.png')

			

			dialog = FlowDiagram(self)
			self.dialogs.append(dialog)
			dialog.setWindowTitle("Classify Example")
			dialog.show()
 
	
		if source.text()=="Reinforcement Learning (DQN)":
			print('Dqn')
			#tetris = Env()
			#agent = DQNAgent(action_size=3)
		else:
			print('Nothing')
		


class FlowDiagram(QMainWindow):

	got_string = QtCore.pyqtSignal(str)

	def __init__(self, parent=None):
		super(FlowDiagram,self).__init__()

		print("here")
		self.labelpic=QLabel()
		pixmap = QPixmap('dotclassify.dot.png')
		self.labelpic.setPixmap(pixmap)
		#self.resize(pixmap.width(),pixmap.height())
		self.trainBut = QPushButton("Train")
			
		self.trainBut.clicked[bool].connect(self.classifyClicked)
		
		self.linkRealBut = QPushButton("Link")
		self.linkRealBut.clicked[bool].connect(self.classifyClicked)

		classifyLayout=QtWidgets.QFormLayout()
		classifyLayout.addWidget(self.labelpic)
		classifyLayout.addWidget(self.trainBut)
		classifyLayout.addWidget(self.linkRealBut)


		self.wid=QWidget(self)
		self.wid.setLayout(classifyLayout)
		self.setCentralWidget(self.wid)

		self.show()

		

	def classifyClicked(self, pressed):
		
		source=self.sender()

		#print(pressed)

		if source.text()=="Train":
			iris= load_iris()
			clf=tree.DecisionTreeClassifier()
			clf=clf.fit(iris.data,iris.target)

			dot_data = tree.export_graphviz(clf, out_file=None) 
			graph = gv.Source(dot_data) 
			
			filename=graph.render("iris") 

			dot_data = tree.export_graphviz(clf, out_file=None, 
				feature_names=iris.feature_names,  
				class_names=iris.target_names,  
				filled=True, rounded=True,  
				special_characters=True)  
			graph = gv.Source(dot_data)
		
			#os.system('dot -Tpng iris.dot -o iris.png')

			graph

			#(graph,)=pydot.graph_from_dot_file('iris.dot')
			#graph.write_png('iris.png') 

			#create new window and show picture
			#dialog =TreeDiagram(self)
			#dialog.setWindowTitle("Tree constructed")
			#dialog.show()
			treeDiag=QMessageBox()
			treeDiag.setWindowTitle("Tree constructed")
			#labelpicTree=QLabel()
			#pixmapTree = QPixmap('iris.png')
			#linkabelpicTree.setPixmap(pixmapTree)
			smaller_pix=QPixmap("iris.png")

			treeDiag.setIconPixmap(smaller_pix)
			#self.resize(pixmap.width(),pixmap.height())
			#treeDiag.addWidget(labelpicTree)
			treeDiag.exec()





			print('predicting value')
			value=clf.predict([[2.,2.,2.,2.]])

			for x in value:
				print(x)

		if source.text()=="Link":
			#attach real data
			print("in link")
			m = DynamicPlotter(sampleinterval=0.05, timewindow=10.)
			m.run()

		else: 
			print("two")
			

	def showTree(self):
		treeDiag=QMessageBox()
		treeDiag.setWindowTitle("Tree constructed")
		labelpicTree=QLabel()
		pixmapTree = QPixmap('iris.png')
		labelpicTree.setPixmap(pixmapTree)

		treeDiag.setIconPixmap(QPixmap("iris.png"))
		#self.resize(pixmap.width(),pixmap.height())
		#treeDiag.addWidget(labelpicTree)
		treeDiag.show()



	
		

class DynamicPlotter():
	def __init__(self, sampleinterval=0.1, timewindow=10.):
		size=(600,350)
		#print("predicted passed value")

		#print(clf.predict([[1.,1.,2.,2.]]))
		self._interval = int(sampleinterval*1000)
		self._bufsize = int(timewindow/sampleinterval)
		self.databuffer = collections.deque([0.0]*self._bufsize, self._bufsize)
		self.x = np.linspace(-timewindow, 0.0, self._bufsize)
		self.y = np.zeros(self._bufsize, dtype=np.int)
		# PyQtGraph stuff
		self.app = QtGui.QApplication([])
		self.plt = pg.plot(title='Dynamic Plotting with PyQtGraph')
		self.plt.resize(*size)
		self.plt.showGrid(x=True, y=True)
		self.plt.setLabel('left', 'Class', 'V')
		self.plt.setLabel('bottom', 'Value', 's')
		self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
		# QTimer
		self.timer = QtCore.QTimer()
		self.timer.timeout.connect(self.updateplot)
		self.timer.start(self._interval)

	def getdata(self):
		frequency = 0.5
		noise = random.normalvariate(0., 1.)
		new = 10.*math.sin(time.time()*frequency*2*math.pi) + noise
		#comment follwoign line for original
		new =random.randint(0, 2)
		print(new)
		return new

	def updateplot(self):
		self.databuffer.append( self.getdata() )
		self.y[:] = self.databuffer
		self.curve.setData(self.x, self.y)
		self.app.processEvents()
		print("update")

	def run(self):
		for x in range(0,1000):
			self.updateplot()
		print("run")





class TreeDiagram(QMainWindow):

	def __init__(self, parent=None):
		super(TreeDiagram,self).__init__()

		print("in treediagram")
		self.labelpicTree=QLabel()
		pixmapTree = QPixmap('iris.png')
		self.labelpicTree.setPixmap(pixmapTree)
		#self.resize(pixmap.width(),pixmap.height())
		treeLayout=QtWidgets.QFormLayout()
		treeLayout.addWidget(self.labelpicTree)

		self.widt=QWidget(self)
		self.widt.setLayout(treeLayout)

		self.setCentralWidget(self.widt)
		self.show()


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	#add icon
	path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'netgameicon.png')
	app.setWindowIcon(QIcon(path)) 
	window = MainWindow()
	window.show()
	sys.exit(app.exec_())



	buttontopo = QPushButton("Load Topology",self)
	buttontopo=setCheckable(True)
	"""	buttontopo.move(10,20)

		

		
	#	buttondata=setCheckable(True)
		buttondata.move(10,60)

		buttondata.clicked[bool].connect(self.buttonclicked)


		buttonDragDrop = QPushButton("Drop Block",self)
	#	buttonDragDrop=setCheckable(True)
		buttonDragDrop.move(10,100)

		buttonDragDrop.clicked[bool].connect(self.buttonclicked)


		buttonYaml = QPushButton("YAML",self)
	#	buttonDragDrop=setCheckable(True)
		buttonYaml.move(10,160)

		buttonYaml.clicked[bool].connect(self.buttonclicked)

		self.square=QFrame(self)
		self.square.setGeometry(150, 20, 300, 300)
		self.square.setStyleSheet("QWidget { background-color: white}")
		"""
"""
		ssh -YC guest@lbl-laser-stacking.dhcp.lbl.gov
(password is 'laserlab')
workon cv
cd ~/work/camera
python live.py
"""