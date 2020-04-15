import os
import sys
import re
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene, QSlider, QGraphicsLineItem
from PyQt5.QtGui import QImage, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QLineF
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
        self.count_lines = set()

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

    def set_lines(self, lines):
        self.count_lines = lines 

    def get_lines(self):
        return self.count_lines

    def __len__(self):
        return self.video_len

class MyGraphicsLineItem(QGraphicsLineItem):
    def __init__(self, x1, y1, x2, y2, pen, parentscene):
        super().__init__(x1, y1, x2, y2)
        self.pen = pen
        self.parentscene = parentscene

    def mousePressEvent(self, event):
        self.parentscene.removeItem(self)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen)
        painter.drawLine(self.line())

class MyGraphicsScene(QGraphicsScene):
    def __init__(self, view):
        super().__init__()
        self.view = view
        self.pixmapitem = None

    def load_image(self, img):
        if self.pixmapitem:
            self.removeItem(self.pixmapitem)
        self.img = img
        # print(img.shape)
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
        # print(event.scenePos().x(), event.scenePos().y())
        pass

class LineSelectGraphicsScene(MyGraphicsScene):
    def __init__(self, view):
        super().__init__(view)
        self.current_line = None
        self.current_pts = None
        self.pen = QPen(Qt.green, 10)
        self.line_list = list()
        self.stage_removal = None

    def mousePressEvent(self, event):
        # print(self.itemAt(event.scenePos().x(), event.scenePos().y()))
        path = QPainterPath()
        path.addRect(event.scenePos().x() - 5, event.scenePos().y() - 5, 10, 10)
        selitems = self.items(path)
        if isinstance(selitems[0], QGraphicsLineItem):
            self.stage_removal = selitems[0]

    def mouseMoveEvent(self, event):
        self.stage_removal = None
        x = event.scenePos().x()
        y = event.scenePos().y()
        if not self.current_line:
            self.current_line = self.addLine(x, y, x, y, self.pen)
            self.current_pts = (x, y)
        else:
            self.current_line.setLine(QLineF(self.current_pts[0], self.current_pts[1], x, y))

        self.update()

    def mouseReleaseEvent(self, event):
        if self.stage_removal:
            self.removeItem(self.stage_removal)
        else:
            self.line_list.append(self.current_line)
            self.current_line = None
        
    def clear_lines(self):
        for line in self.line_list:
            self.removeItem(line)
        self.line_list = list()

    def draw_lines(self, lines):
        self.clear_lines()
        for line in lines:
            self.line_list.append(self.addLine(line[0], line[1], line[2], line[3], self.pen))

    def get_lines(self):
        return [[line.line().x1(), line.line().y1(), line.line().x2(), line.line().y2()] for line in self.line_list]

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
        self.graphicsScene_main = LineSelectGraphicsScene(self.graphicsView_main)
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
        self.prev_vid_index = -1

    def change_vid_index(self, nind):
        self.prev_vid_index = self.vid_index
        self.vid_index = nind

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
        # print(self.vid_list)
        if len(self.vid_list) > 1:
            self.pushButton_next_cam.setDisabled(False)
        self.horizontalSlider.setDisabled(False)
        self.radioButton_lines.setDisabled(False)
        self.radioButton_map.setDisabled(False)
        self.change_vid_index(0)
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
        # print(self.horizontalSlider.value())
        self.graphicsScene_main.load_image(self.vid_list[self.vid_index].getFrame(self.horizontalSlider.value()))

    def next_button_click(self):
        self.change_vid_index(self.vid_index + 1)
        if self.vid_index == (len(self.vid_list) - 1):
            self.pushButton_next_cam.setDisabled(True)
        self.pushButton_prev_cam.setDisabled(False)
        self.update_view()
    
    def prev_button_click(self):
        self.change_vid_index(self.vid_index - 1)
        if self.vid_index == 0:
            self.pushButton_prev_cam.setDisabled(True)
        self.pushButton_next_cam.setDisabled(False)
        self.update_view()

    def update_view(self):
        # print(self.prev_vid_index)
        self.vid_list[self.prev_vid_index].set_lines(self.graphicsScene_main.get_lines())

        vid = self.vid_list[self.vid_index]
        self.horizontalSlider.setMaximum(100)
        self.horizontalSlider.setValue(vid.getCurrentFrameIndex())
        self.graphicsScene_main.load_image(vid.getCurrentFrame())
        self.graphicsScene_main.draw_lines(vid.get_lines())
            

if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = Consumer()

    currentForm.show()
    currentApp.exec_()
