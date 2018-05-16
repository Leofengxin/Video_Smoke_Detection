import os
import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


def newIcon(icon_name):
    icon_path = os.path.join(os.path.dirname(os.getcwd()), 'win_icons', icon_name)
    return QIcon(icon_path)

def newAction(parent, text, slot=None, shortcut=None,
              icon=None, tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    action = QAction(text, parent)
    if icon is not None:
        action.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            action.setShortcuts(shortcut)
        else:
            action.setShortcut(shortcut)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if slot is not None:
        action.triggered.connect(slot)
    action.setCheckable(checkable)
    action.setEnabled(enabled)
    return action

def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)

def cv2_img_to_qt_pixmap(cv2_img):
    height, width, bytesPerComponent = cv2_img.shape
    bytes_per_line = 3 * width
    cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB, cv2_img)
    qt_img = QImage(cv2_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    qt_pixmap = QPixmap.fromImage(qt_img)
    return qt_pixmap

class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class VideoInfo(object):
    def __init__(self):
        self.video_path = None
        self.frame_rate = 1
        self.frame_total_num = 0
        self.img_height = 0
        self.img_width = 0
        self.frame_current = 0