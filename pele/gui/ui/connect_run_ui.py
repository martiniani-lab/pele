# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect_run_ui.ui'
#
# Created: Tue Oct 22 11:03:50 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from __future__ import absolute_import
from builtins import object
from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(764, 829)
        MainWindow.setDockNestingEnabled(False)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 764, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.view_Log = QtGui.QDockWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.view_Log.sizePolicy().hasHeightForWidth()
        )
        self.view_Log.setSizePolicy(sizePolicy)
        self.view_Log.setFeatures(
            QtGui.QDockWidget.DockWidgetFloatable
            | QtGui.QDockWidget.DockWidgetMovable
        )
        self.view_Log.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.view_Log.setObjectName(_fromUtf8("view_Log"))
        self.dockWidgetContents_5 = QtGui.QWidget()
        self.dockWidgetContents_5.setObjectName(
            _fromUtf8("dockWidgetContents_5")
        )
        self.horizontalLayout = QtGui.QHBoxLayout(self.dockWidgetContents_5)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.textEdit = QtGui.QTextEdit(self.dockWidgetContents_5)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.textEdit.sizePolicy().hasHeightForWidth()
        )
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setMinimumSize(QtCore.QSize(0, 0))
        self.textEdit.setBaseSize(QtCore.QSize(0, 500))
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.horizontalLayout.addWidget(self.textEdit)
        self.view_Log.setWidget(self.dockWidgetContents_5)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.view_Log)
        self.view_ogl = QtGui.QDockWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.view_ogl.sizePolicy().hasHeightForWidth()
        )
        self.view_ogl.setSizePolicy(sizePolicy)
        self.view_ogl.setFeatures(
            QtGui.QDockWidget.DockWidgetFloatable
            | QtGui.QDockWidget.DockWidgetMovable
        )
        self.view_ogl.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.view_ogl.setObjectName(_fromUtf8("view_ogl"))
        self.dockWidgetContents_3 = QtGui.QWidget()
        self.dockWidgetContents_3.setObjectName(
            _fromUtf8("dockWidgetContents_3")
        )
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.dockWidgetContents_3)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.ogl = Show3DWithSlider(self.dockWidgetContents_3)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ogl.sizePolicy().hasHeightForWidth())
        self.ogl.setSizePolicy(sizePolicy)
        self.ogl.setObjectName(_fromUtf8("ogl"))
        self.horizontalLayout_2.addWidget(self.ogl)
        self.view_ogl.setWidget(self.dockWidgetContents_3)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(4), self.view_ogl)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionEnergy = QtGui.QAction(MainWindow)
        self.actionEnergy.setCheckable(True)
        self.actionEnergy.setChecked(False)
        self.actionEnergy.setObjectName(_fromUtf8("actionEnergy"))
        self.action3D = QtGui.QAction(MainWindow)
        self.action3D.setCheckable(True)
        self.action3D.setObjectName(_fromUtf8("action3D"))
        self.actionGraph = QtGui.QAction(MainWindow)
        self.actionGraph.setCheckable(True)
        self.actionGraph.setObjectName(_fromUtf8("actionGraph"))
        self.actionPause = QtGui.QAction(MainWindow)
        self.actionPause.setCheckable(True)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap(_fromUtf8(":/icons/icons/pause.png")),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )
        icon.addPixmap(
            QtGui.QPixmap(_fromUtf8(":/icons/icons/pause.png")),
            QtGui.QIcon.Normal,
            QtGui.QIcon.On,
        )
        self.actionPause.setIcon(icon)
        self.actionPause.setObjectName(_fromUtf8("actionPause"))
        self.actionD_Graph = QtGui.QAction(MainWindow)
        self.actionD_Graph.setCheckable(True)
        self.actionD_Graph.setObjectName(_fromUtf8("actionD_Graph"))
        self.actionSummary = QtGui.QAction(MainWindow)
        self.actionSummary.setCheckable(True)
        self.actionSummary.setObjectName(_fromUtf8("actionSummary"))
        self.actionLog = QtGui.QAction(MainWindow)
        self.actionLog.setCheckable(True)
        self.actionLog.setObjectName(_fromUtf8("actionLog"))
        self.actionKill = QtGui.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap(_fromUtf8(":/icons/icons/stop.png")),
            QtGui.QIcon.Normal,
            QtGui.QIcon.Off,
        )
        self.actionKill.setIcon(icon1)
        self.actionKill.setObjectName(_fromUtf8("actionKill"))
        self.toolBar.addAction(self.actionPause)
        self.toolBar.addAction(self.actionKill)
        self.toolBar.addAction(self.actionLog)
        self.toolBar.addAction(self.action3D)
        self.toolBar.addAction(self.actionEnergy)
        self.toolBar.addAction(self.actionGraph)
        self.toolBar.addAction(self.actionD_Graph)
        self.toolBar.addAction(self.actionSummary)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.view_Log.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Log", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.view_ogl.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "3D view", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.toolBar.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionEnergy.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Energy", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionEnergy.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "toggle energy window",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.action3D.setText(
            QtGui.QApplication.translate(
                "MainWindow", "3D", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.action3D.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "toggle 3D viewer",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionGraph.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Graph", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionGraph.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "toggle graph view",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionPause.setText(
            QtGui.QApplication.translate(
                "MainWindow", "pause", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionPause.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "pause / restart",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionD_Graph.setText(
            QtGui.QApplication.translate(
                "MainWindow", "D-Graph", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionD_Graph.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "disconnectivity graph",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionSummary.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Summary", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionSummary.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "display summary information",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionLog.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Log", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionLog.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "display log information",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionKill.setText(
            QtGui.QApplication.translate(
                "MainWindow", "kill", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionKill.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "interrupt the connect run",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )


from pele.gui.show3d_with_slider import Show3DWithSlider
from . import resources_rc
