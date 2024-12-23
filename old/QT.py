import subprocess
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QLineEdit, QMainWindow
from PyQt5.uic import loadUi
# import mbbbean
import pto

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("1window.ui", self)
        self.bSIMP.clicked.connect(self.gotoSIMPTasksScreen)
        self.bPTO.clicked.connect(self.gotoPTOTasksScreen)

    def gotoSIMPTasksScreen(self):
        simpTasksScreen = SIMPTasksScreen()
        widget.addWidget(simpTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoPTOTasksScreen(self):
        ptoTasksScreen = PTOTasksScreen()
        widget.addWidget(ptoTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class SIMPTasksScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("2window.ui", self)
        self.buExit.clicked.connect(self.gotoMainWindow)
        self.b1.clicked.connect(self.gotoSIMPmbbTaskScreen)
        self.b2.clicked.connect(self.gotoSIMPcantTaskScreen)
        self.b3.clicked.connect(self.gotoSIMPLbracTaskScreen)

    def gotoMainWindow(self):
        mainwindow = MainWindow()
        widget.addWidget(mainwindow)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoSIMPmbbTaskScreen(self):
        simpmbbTaskScreen = SIMPmbbTaskScreen()
        widget.addWidget(simpmbbTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoSIMPcantTaskScreen(self):
        simpcantTaskScreen = SIMPcantTaskScreen()
        widget.addWidget(simpcantTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoSIMPLbracTaskScreen(self):
        simpLbracTaskScreen = SIMPLbracTaskScreen()
        widget.addWidget(simpLbracTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class SIMPmbbTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("3window.ui", self)
        self.buExit.clicked.connect(self.gotoSIMPTasksScreen)
        self.buRun.clicked.connect(self.run_SIMPmbb)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoSIMPTasksScreen(self):
        simpTasksScreen = SIMPTasksScreen()
        widget.addWidget(simpTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_SIMPmbb(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python SIMPmbb.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

class PTOTasksScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("4window.ui", self)
        self.buExit.clicked.connect(self.gotoMainWindow)
        self.b1.clicked.connect(self.gotoPTOmbbTaskScreen)
        self.b2.clicked.connect(self.gotoPTOcantTaskScreen)
        self.b3.clicked.connect(self.gotoPTOLbracTaskScreen)

    def gotoMainWindow(self):
        mainwindow = MainWindow()
        widget.addWidget(mainwindow)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoPTOmbbTaskScreen(self):
        ptombbTaskScreen = PTOmbbTaskScreen()
        widget.addWidget(ptombbTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoPTOcantTaskScreen(self):
        ptocantTaskScreen = PTOcantTaskScreen()
        widget.addWidget(ptocantTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def gotoPTOLbracTaskScreen(self):
        ptoLbracTaskScreen = PTOLbracTaskScreen()
        widget.addWidget(ptoLbracTaskScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

class PTOmbbTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("5window.ui", self)
        self.buExit.clicked.connect(self.gotoPTOTasksScreen)
        self.buRun.clicked.connect(self.run_PTOsmbb)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoPTOTasksScreen(self):
        ptoTasksScreen = PTOTasksScreen()
        widget.addWidget(ptoTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_PTOsmbb(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python PTOsmbb.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

class SIMPcantTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("6window.ui", self)
        self.buExit.clicked.connect(self.gotoSIMPTasksScreen)
        self.buRun1.clicked.connect(self.run_SIMP1load)
        self.buRun.clicked.connect(self.run_SIMPcant)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)
        self.combo1.addItem('Сталь', 0.3)
        self.combo1.addItem('Чугун', 0.25)
        self.combo1.addItem('Медь', 0.32)
        self.combo1.addItem('Бетон', 0.16)
        self.combo1.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoSIMPTasksScreen(self):
        simpTasksScreen = SIMPTasksScreen()
        widget.addWidget(simpTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_SIMPcant(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python SIMPcant.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

    def run_SIMP1load(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght1")
        leWidth = self.findChild(QLineEdit, "lineEditWidth1")
        subprocess.call('python SIMP1load.py {0} {1} {2}'.format(float(self.combo.currentData()), leLenght.text(), leWidth.text()), shell=True)

class SIMPLbracTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("7window.ui", self)
        self.buExit.clicked.connect(self.gotoSIMPTasksScreen)
        self.buRun.clicked.connect(self.run_SIMPLbrac)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoSIMPTasksScreen(self):
        simpTasksScreen = SIMPTasksScreen()
        widget.addWidget(simpTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_SIMPLbrac(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python SIMPLbrac.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

class PTOcantTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("8window.ui", self)
        self.buExit.clicked.connect(self.gotoPTOTasksScreen)
        self.buRun.clicked.connect(self.run_PTOcant)
        self.buRun1.clicked.connect(self.run_PTO1load)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)
        self.combo1.addItem('Сталь', 0.3)
        self.combo1.addItem('Чугун', 0.25)
        self.combo1.addItem('Медь', 0.32)
        self.combo1.addItem('Бетон', 0.16)
        self.combo1.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoPTOTasksScreen(self):
        ptoTasksScreen = PTOTasksScreen()
        widget.addWidget(ptoTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_PTOcant(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python PTOscant.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

    def run_PTO1load(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght1")
        leWidth = self.findChild(QLineEdit, "lineEditWidth1")
        subprocess.call('python PTOs1load.py {0} {1} {2}'.format(float(self.combo.currentData()), leLenght.text(), leWidth.text()), shell=True)

class PTOLbracTaskScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("9window.ui", self)
        self.buExit.clicked.connect(self.gotoPTOTasksScreen)
        self.buRun.clicked.connect(self.run_PTOsLbrac)
        self.combo.addItem('Сталь', 0.3)
        self.combo.addItem('Чугун', 0.25)
        self.combo.addItem('Медь', 0.32)
        self.combo.addItem('Бетон', 0.16)
        self.combo.activated.connect(self.onActivated)

    def onActivated(self, index):
        nu = self.combo.itemData(index)
        return nu

    def gotoPTOTasksScreen(self):
        ptoTasksScreen = PTOTasksScreen()
        widget.addWidget(ptoTasksScreen)
        widget.setCurrentIndex(widget.currentIndex() + 1)

    def run_PTOsLbrac(self):
        leLenght = self.findChild(QLineEdit, "lineEditLenght")
        subprocess.call('python PTOsLbrac.py {0} {1}'.format(float(self.combo.currentData()), leLenght.text()), shell=True)

app = QApplication([])
widget = QtWidgets.QStackedWidget()
mainwindow = MainWindow()
widget.addWidget(mainwindow)
widget.setFixedHeight(895)
widget.setFixedWidth(1093)
widget.show()
