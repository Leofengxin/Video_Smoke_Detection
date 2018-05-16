from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class Canvas(QWidget):
    def __init__(self):
        super(Canvas, self).__init__()
        self.pixmap = QPixmap()
        self.smoke_blocks = []
        self.motion_blocks = []
        self._painter = QPainter()
        self.is_hide_smoke_blocks = False
        self.is_hide_motion_blocks = False

    def paintEvent(self, QPaintEvent):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(QPaintEvent)

        self.setMinimumSize(self.pixmap.size())
        painter = self._painter
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # Draw pixmap.
        painter.drawPixmap(0, 0, self.pixmap)
        # Draw motion_blocks and smoke_blocks.
        if self.is_hide_motion_blocks is False:
            color = QColor(255, 0, 0, 255)
            painter.setPen(QPen(color))
            for m_block in self.motion_blocks:
                y, x, w, h = m_block
                painter.drawRect(x, y, w, h)
        if self.is_hide_smoke_blocks is False:
            color = QColor(0, 255, 0, 255)
            painter.setPen(QPen(color))
            for s_block in self.smoke_blocks:
                y, x, w, h = s_block
                painter.drawEllipse(x, y, w, h)
        painter.end()

    def load_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.repaint()