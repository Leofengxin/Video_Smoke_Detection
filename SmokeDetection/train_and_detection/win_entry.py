from logging_core.logging_setup import setup_logging
setup_logging()
import sys
from PyQt5.QtWidgets import QApplication
from win_libs.libs_auxiliary import newIcon
from win_libs.main_win import MainWindow

def smoke_detection_win(sess=None, model=None, appname='train_mode'):
    app = QApplication(sys.argv)
    app.setApplicationName(appname)
    app.setWindowIcon(newIcon('app.png'))
    app.setStyle('fusion')
    win = MainWindow(sess, model)
    win.show()
    flag = app.exec_()
    return flag

if __name__ == '__main__':
    appname = 'img_smoke_detection'
    flag = smoke_detection_win(appname=appname)
    sys.exit(flag)