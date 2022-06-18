import sys

import cv2
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap


class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        print("Starting video streaming")
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        cap.release()

    def stop(self):
        print("Stopping streaming")
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self, display_width, display_height):
        super().__init__()
        self.setWindowTitle("Qt webcam")
        self.display_width = display_width
        self.display_height = display_height
        self.image = QLabel(self)
        self.image.resize(self.display_width, self.display_height)

        self.add_buttons()
        self.set_layout()

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, QtCore.Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def add_buttons(self):
        self.quit_app_button = QtWidgets.QPushButton("Quit app")
        self.quit_app_button.clicked.connect(self.close_app)

    def set_layout(self):
        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(self.quit_app_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(buttons_layout)
        layout.addWidget(self.image)

        self.setLayout(layout)

    def close_app(self):
        question = QtWidgets.QMessageBox.question(self, 'Extract', 'Do you really want to quit?',
                                                  QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if question == QtWidgets.QMessageBox.Yes:
            print("Closing app")
            self.video_thread.stop()
            cv2.destroyAllWindows()
            sys.exit()
