import qdarkstyle
import pyqtgraph as pg
from pyqtgraph.widgets.TableWidget import TableWidget
from FRCM import Ui_MainWindow
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from qt_material import apply_stylesheet
import joblib
model=joblib.load('TrAGBDT.model')

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        #设置字体
        self.font=QFont("Times New Roman",10)
        self.configCombo.setFont(self.font)
        self.config.setFont(self.font)
        self.fc.setFont(self.font)
        self.lineEdit_fc.setFont(self.font)
        self.ad.setFont(self.font)
        self.lineEdit_ad.setFont(self.font)
        self.rhof.setFont(self.font)
        self.lineEdit_rhof.setFont(self.font)
        self.bw.setFont(self.font)
        self.lineEdit_bw.setFont(self.font)
        self.rhosy.setFont(self.font)
        self.lineEdit_rhosy.setFont(self.font)
        self.Es.setFont(self.font)
        self.lineEdit_Es.setFont(self.font)
        self.fu.setFont(self.font)
        self.lineEdit_fu.setFont(self.font)
        self.fsy.setFont(self.font)
        self.lineEdit_fsy.setFont(self.font)
        self.Vf.setFont(self.font)
        self.lineEdit_Vf.setFont(self.font)
        self.start.setFont(self.font)

        #设置默认值
        self.lineEdit_bw.setText(str(150))
        self.lineEdit_ad.setText(str(2.85))
        self.lineEdit_fc.setText(str(30.61))
        self.lineEdit_rhosy.setText(str(0.14))
        self.lineEdit_fsy.setText(str(275))
        self.lineEdit_Es.setText(str(225))
        self.lineEdit_fu.setText(str(3350))
        self.lineEdit_rhof.setText(str(1.3))
        
    @pyqtSlot()
    def on_start_clicked(self):
        bw=float(self.lineEdit_bw.text())
        ad=float(self.lineEdit_ad.text())
        fc=float(self.lineEdit_fc.text())
        rhosy=float(self.lineEdit_rhosy.text())
        rhof=float(self.lineEdit_rhof.text())
        fsy=float(self.lineEdit_fsy.text())
        Ef=float(self.lineEdit_Es.text())
        fu=float(self.lineEdit_fu.text())
        config=self.configCombo.currentText()
        if config=='FW,CN':
            config=3
        elif config=='SB,CN':
            config=1
        elif config=='UW,CN':
            config=2
        else:
            config=0
        x=np.array([bw,ad,fc,rhosy,fsy,Ef,fu,config,rhof]).reshape(1,-1)
        Vf=model.predict(x)[0]
        self.lineEdit_Vf.setText(str(round(Vf,1)))
        
        

app =QApplication(sys.argv)
apply_stylesheet(app, theme='dark_teal.xml')
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
window = MainWindow()
window.show()
