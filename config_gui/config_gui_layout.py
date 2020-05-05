# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'config_gui_layout.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2047, 1637)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_vid_dir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_vid_dir.setGeometry(QtCore.QRect(30, 40, 301, 49))
        self.pushButton_vid_dir.setObjectName("pushButton_vid_dir")
        self.graphicsView_main = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_main.setGeometry(QtCore.QRect(30, 110, 1981, 1251))
        self.graphicsView_main.setObjectName("graphicsView_main")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(30, 1380, 1981, 31))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.pushButton_prev_cam = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_prev_cam.setGeometry(QtCore.QRect(800, 1480, 231, 49))
        self.pushButton_prev_cam.setObjectName("pushButton_prev_cam")
        self.pushButton_next_cam = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_next_cam.setGeometry(QtCore.QRect(1040, 1480, 231, 49))
        self.pushButton_next_cam.setObjectName("pushButton_next_cam")
        self.radioButton_lines = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_lines.setGeometry(QtCore.QRect(110, 1460, 193, 41))
        self.radioButton_lines.setObjectName("radioButton_lines")
        self.radioButton_map = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_map.setGeometry(QtCore.QRect(110, 1500, 251, 41))
        self.radioButton_map.setObjectName("radioButton_map")
        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(1860, 1480, 153, 49))
        self.pushButton_save.setObjectName("pushButton_save")
        self.graphicsView_sub_left = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_sub_left.setGeometry(QtCore.QRect(30, 370, 971, 621))
        self.graphicsView_sub_left.setObjectName("graphicsView_sub_left")
        self.graphicsView_sub_right = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_sub_right.setGeometry(QtCore.QRect(1040, 370, 971, 621))
        self.graphicsView_sub_right.setObjectName("graphicsView_sub_right")
        self.pushButton_choose_map = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_choose_map.setGeometry(QtCore.QRect(350, 40, 251, 49))
        self.pushButton_choose_map.setObjectName("pushButton_choose_map")
        self.lineEdit_slider = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_slider.setGeometry(QtCore.QRect(970, 1410, 113, 43))
        self.lineEdit_slider.setObjectName("lineEdit_slider")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2047, 40))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_vid_dir.setText(_translate("MainWindow", "Choose Video Directory"))
        self.pushButton_prev_cam.setText(_translate("MainWindow", "Prev Camera"))
        self.pushButton_next_cam.setText(_translate("MainWindow", "Next Camera"))
        self.radioButton_lines.setText(_translate("MainWindow", "line trigger"))
        self.radioButton_map.setText(_translate("MainWindow", "mapping points"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.pushButton_choose_map.setText(_translate("MainWindow", "Choose Map"))

