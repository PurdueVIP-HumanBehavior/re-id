import os
import sys
import re
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QSlider
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor
from PyQt5.QtCore import Qt, QRectF
import numpy as np
import cv2
from config_gui_layout import *

class CamItem:
    def __init__(self, path):
        self.path = path
        self.frame = 0

    def getFrame(self, frame): raise NotImplementedError("Override me")

    def update_view(self, slider, scene_list):
        pass

class VideoCamItem(CamItem):
    def __init__(self, path):
        super().__init__(path)
        self.video = cv2.VideoCapture(self.path)
        self.video_len = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval = int(self.video_len/100)
        self.frame_cache = dict()
        self.frame = 0

    def getFrame(self, frame):
        nframe = frame * self.frame_interval
        if nframe >= self.video_len: nframe = self.video_len - 1

        if nframe < 0 or nframe >= self.video_len:
            return None
        if frame not in self.frame_cache:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, nframe)
            success, retval = self.video.read()
            if success: self.frame_cache[frame] = cv2.cvtColor(retval, cv2.COLOR_BGR2RGB)
            else: return None

        self.frame = frame
        return self.frame_cache[frame]

    def getCurrentFrame(self):
        return self.getFrame(self.frame)

    def getCurrentFrameIndex(self):
        return self.frame

    def __len__(self):
        return self.video_len

class MyGraphicsScene(QGraphicsScene):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.pixmapitem = None

    def load_image(self, img):
        if self.pixmapitem:
            self.removeItem(self.pixmapitem)
        self.img = img
        print(img.shape)
        if len(self.img.shape) > 2:
            h, w, _ = self.img.shape
            self.imgq = QImage(self.img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            h, w = self.img.shape
            self.imgq = QImage(self.img.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(self.imgq)
        self.pixmapitem = self.addPixmap(pixmap)

        self.view.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self.update()

    def mousePressEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        print(event.scenePos().x(), event.scenePos().y())

class Consumer(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Consumer, self).__init__(parent)
        self.setupUi(self)

        # choose file buttons setup
        self.pushButton_vid_dir.clicked.connect(self.load_video_directory)
        self.pushButton_choose_map.clicked.connect(self.load_map)

        # nav buttons setup
        self.pushButton_prev_cam.setDisabled(True)
        self.pushButton_prev_cam.clicked.connect(self.prev_button_click)
        self.pushButton_next_cam.setDisabled(True)
        self.pushButton_next_cam.clicked.connect(self.next_button_click)

        # radio button setups
        self.radioButton_lines.toggle()
        self.graphicsView_sub_left.setVisible(False)
        self.graphicsView_sub_right.setVisible(False)
        self.radioButton_lines.toggled.connect(self.toggle_mode)
        self.radioButton_lines.setDisabled(True)
        self.radioButton_map.setDisabled(True)
        
        #set up views
        self.graphicsScene_main = MyGraphicsScene(self.graphicsView_main)
        self.graphicsView_main.setScene(self.graphicsScene_main)

        self.graphicsScene_sub_left = MyGraphicsScene(self.graphicsView_sub_left)
        self.graphicsView_sub_left.setScene(self.graphicsScene_sub_left)

        self.graphicsScene_sub_right = MyGraphicsScene(self.graphicsView_sub_right)
        self.graphicsView_sub_right.setScene(self.graphicsScene_sub_right)

        # slider
        self.horizontalSlider.setDisabled(True)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setTracking(False)
        self.horizontalSlider.valueChanged.connect(self.hoiz_slider_val_changed)

        self.vid_list = list()
        self.vid_index = -1

    def get_image_filename(self):

        filePath, _ = QFileDialog.getOpenFileName(self, caption='Open Image file ...', filter="Images (*.png *.xpm *.jpg)")

        if not filePath:
            return None

        return filePath

    def get_directory_name(self):

        dirname = QFileDialog.getExistingDirectory(self, caption='Select Video Directory')

        if not dirname:
            return None

        return dirname 
        
    def load_video_directory(self):
        # dirname = self.get_directory_name()
        self.vid_list = list()
        dirname = "/media/moiz/Windows/Users/Moiz/Documents/CAM2/reid-data/bidc2/" 
        vals = os.listdir(dirname)
        for item in vals:
            if item.endswith('.MOV') or item.endswith('.avi') or item.endswith('.mp4'):
                self.vid_list.append(VideoCamItem(os.path.join(dirname, item)))
        print(self.vid_list)
        if len(self.vid_list) > 1:
            self.pushButton_next_cam.setDisabled(False)
        self.horizontalSlider.setDisabled(False)
        self.radioButton_lines.setDisabled(False)
        self.radioButton_map.setDisabled(False)
        self.vid_index = 0
        self.update_view()

    def load_map(self):
        map_filename = self.get_image_filename()
        print(map_filename)

    def toggle_mode(self):
        if self.radioButton_lines.isChecked():
            self.graphicsView_sub_left.setVisible(False)
            self.graphicsView_sub_right.setVisible(False)
            self.graphicsView_main.setVisible(True)
        else:
            self.graphicsView_main.setVisible(False)
            self.graphicsView_sub_left.setVisible(True)
            self.graphicsView_sub_right.setVisible(True)

    def hoiz_slider_val_changed(self):
        print(self.horizontalSlider.value())
        self.graphicsScene_main.load_image(self.vid_list[self.vid_index].getFrame(self.horizontalSlider.value()))

    def next_button_click(self):
        self.vid_index += 1
        if self.vid_index == (len(self.vid_list) - 1):
            self.pushButton_next_cam.setDisabled(True)
        self.pushButton_prev_cam.setDisabled(False)
        self.update_view()
    
    def prev_button_click(self):
        self.vid_index -= 1
        if self.vid_index == 0:
            self.pushButton_prev_cam.setDisabled(True)
        self.pushButton_next_cam.setDisabled(False)
        self.update_view()

    def update_view(self):
        vid = self.vid_list[self.vid_index]
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setValue(vid.getCurrentFrameIndex())
        self.graphicsScene_main.load_image(vid.getCurrentFrame())
            

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = Consumer()

    currentForm.show()
    currentApp.exec_()
