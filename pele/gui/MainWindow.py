# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created: Mon Feb 17 17:24:28 2014
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
        MainWindow.resize(839, 623)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            MainWindow.sizePolicy().hasHeightForWidth()
        )
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.BHTab = QtGui.QWidget()
        self.BHTab.setObjectName(_fromUtf8("BHTab"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.BHTab)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.ogl_main = Show3D(self.BHTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.ogl_main.sizePolicy().hasHeightForWidth()
        )
        self.ogl_main.setSizePolicy(sizePolicy)
        self.ogl_main.setObjectName(_fromUtf8("ogl_main"))
        self.horizontalLayout_3.addWidget(self.ogl_main)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.list_minima_main = QtGui.QTableView(self.BHTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.list_minima_main.sizePolicy().hasHeightForWidth()
        )
        self.list_minima_main.setSizePolicy(sizePolicy)
        self.list_minima_main.setMaximumSize(QtCore.QSize(200, 16777215))
        self.list_minima_main.setEditTriggers(
            QtGui.QAbstractItemView.NoEditTriggers
        )
        self.list_minima_main.setSelectionMode(
            QtGui.QAbstractItemView.SingleSelection
        )
        self.list_minima_main.setSortingEnabled(True)
        self.list_minima_main.setObjectName(_fromUtf8("list_minima_main"))
        self.list_minima_main.horizontalHeader().setCascadingSectionResizes(
            True
        )
        self.list_minima_main.horizontalHeader().setDefaultSectionSize(80)
        self.list_minima_main.horizontalHeader().setSortIndicatorShown(True)
        self.list_minima_main.horizontalHeader().setStretchLastSection(False)
        self.list_minima_main.verticalHeader().setVisible(False)
        self.list_minima_main.verticalHeader().setCascadingSectionResizes(False)
        self.list_minima_main.verticalHeader().setSortIndicatorShown(False)
        self.verticalLayout_3.addWidget(self.list_minima_main)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setContentsMargins(-1, 10, -1, -1)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.btn_stop_basinhopping = QtGui.QPushButton(self.BHTab)
        self.btn_stop_basinhopping.setObjectName(
            _fromUtf8("btn_stop_basinhopping")
        )
        self.gridLayout_3.addWidget(self.btn_stop_basinhopping, 4, 0, 1, 1)
        self.pushNormalmodesMin = QtGui.QPushButton(self.BHTab)
        self.pushNormalmodesMin.setObjectName(_fromUtf8("pushNormalmodesMin"))
        self.gridLayout_3.addWidget(self.pushNormalmodesMin, 0, 1, 1, 1)
        self.pushTakestepExplorer = QtGui.QPushButton(self.BHTab)
        self.pushTakestepExplorer.setObjectName(
            _fromUtf8("pushTakestepExplorer")
        )
        self.gridLayout_3.addWidget(self.pushTakestepExplorer, 2, 0, 1, 1)
        self.btn_heat_capacity = QtGui.QPushButton(self.BHTab)
        self.btn_heat_capacity.setObjectName(_fromUtf8("btn_heat_capacity"))
        self.gridLayout_3.addWidget(self.btn_heat_capacity, 0, 0, 1, 1)
        self.btn_start_basinhopping = QtGui.QPushButton(self.BHTab)
        self.btn_start_basinhopping.setObjectName(
            _fromUtf8("btn_start_basinhopping")
        )
        self.gridLayout_3.addWidget(self.btn_start_basinhopping, 3, 0, 1, 1)
        self.label_bh_nproc = QtGui.QLabel(self.BHTab)
        self.label_bh_nproc.setObjectName(_fromUtf8("label_bh_nproc"))
        self.gridLayout_3.addWidget(self.label_bh_nproc, 4, 1, 1, 1)
        self.lineEdit_bh_nsteps = QtGui.QLineEdit(self.BHTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.lineEdit_bh_nsteps.sizePolicy().hasHeightForWidth()
        )
        self.lineEdit_bh_nsteps.setSizePolicy(sizePolicy)
        self.lineEdit_bh_nsteps.setObjectName(_fromUtf8("lineEdit_bh_nsteps"))
        self.gridLayout_3.addWidget(self.lineEdit_bh_nsteps, 3, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_3)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.tabWidget.addTab(self.BHTab, _fromUtf8(""))
        self.NEBTab = QtGui.QWidget()
        self.NEBTab.setObjectName(_fromUtf8("NEBTab"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.NEBTab)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, -1, -1, -1)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.oglPath = Show3DWithSlider(self.NEBTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.oglPath.sizePolicy().hasHeightForWidth()
        )
        self.oglPath.setSizePolicy(sizePolicy)
        self.oglPath.setObjectName(_fromUtf8("oglPath"))
        self.verticalLayout.addWidget(self.oglPath)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(6, -1, -1, -1)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.listMinima1 = QtGui.QTableView(self.NEBTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.listMinima1.sizePolicy().hasHeightForWidth()
        )
        self.listMinima1.setSizePolicy(sizePolicy)
        self.listMinima1.setMaximumSize(QtCore.QSize(200, 16777215))
        self.listMinima1.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.listMinima1.setSelectionMode(
            QtGui.QAbstractItemView.SingleSelection
        )
        self.listMinima1.setSortingEnabled(True)
        self.listMinima1.setObjectName(_fromUtf8("listMinima1"))
        self.listMinima1.horizontalHeader().setDefaultSectionSize(85)
        self.listMinima1.verticalHeader().setVisible(False)
        self.verticalLayout_5.addWidget(self.listMinima1)
        self.listMinima2 = QtGui.QTableView(self.NEBTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.listMinima2.sizePolicy().hasHeightForWidth()
        )
        self.listMinima2.setSizePolicy(sizePolicy)
        self.listMinima2.setMaximumSize(QtCore.QSize(200, 16777215))
        self.listMinima2.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.listMinima2.setSelectionMode(
            QtGui.QAbstractItemView.SingleSelection
        )
        self.listMinima2.setSortingEnabled(True)
        self.listMinima2.setObjectName(_fromUtf8("listMinima2"))
        self.listMinima2.horizontalHeader().setDefaultSectionSize(80)
        self.listMinima2.verticalHeader().setVisible(False)
        self.verticalLayout_5.addWidget(self.listMinima2)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setContentsMargins(20, 0, 20, -1)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.btnAlign = QtGui.QPushButton(self.NEBTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.btnAlign.sizePolicy().hasHeightForWidth()
        )
        self.btnAlign.setSizePolicy(sizePolicy)
        self.btnAlign.setMaximumSize(QtCore.QSize(100, 100))
        self.btnAlign.setObjectName(_fromUtf8("btnAlign"))
        self.gridLayout_2.addWidget(self.btnAlign, 1, 0, 1, 1)
        self.btnReconnect = QtGui.QPushButton(self.NEBTab)
        self.btnReconnect.setObjectName(_fromUtf8("btnReconnect"))
        self.gridLayout_2.addWidget(self.btnReconnect, 3, 1, 1, 1)
        self.btnNEB = QtGui.QPushButton(self.NEBTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.btnNEB.sizePolicy().hasHeightForWidth()
        )
        self.btnNEB.setSizePolicy(sizePolicy)
        self.btnNEB.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btnNEB.setObjectName(_fromUtf8("btnNEB"))
        self.gridLayout_2.addWidget(self.btnNEB, 2, 0, 1, 1)
        self.btnConnect = QtGui.QPushButton(self.NEBTab)
        self.btnConnect.setObjectName(_fromUtf8("btnConnect"))
        self.gridLayout_2.addWidget(self.btnConnect, 3, 0, 1, 1)
        self.btnShowGraph = QtGui.QPushButton(self.NEBTab)
        self.btnShowGraph.setObjectName(_fromUtf8("btnShowGraph"))
        self.gridLayout_2.addWidget(self.btnShowGraph, 5, 0, 1, 1)
        self.btnDisconnectivity_graph = QtGui.QPushButton(self.NEBTab)
        self.btnDisconnectivity_graph.setObjectName(
            _fromUtf8("btnDisconnectivity_graph")
        )
        self.gridLayout_2.addWidget(self.btnDisconnectivity_graph, 5, 1, 1, 1)
        self.btn_connect_in_optim = QtGui.QPushButton(self.NEBTab)
        self.btn_connect_in_optim.setObjectName(
            _fromUtf8("btn_connect_in_optim")
        )
        self.gridLayout_2.addWidget(self.btn_connect_in_optim, 4, 1, 1, 1)
        self.btn_connect_all = QtGui.QPushButton(self.NEBTab)
        self.btn_connect_all.setObjectName(_fromUtf8("btn_connect_all"))
        self.gridLayout_2.addWidget(self.btn_connect_all, 4, 0, 1, 1)
        self.btn_close_all = QtGui.QPushButton(self.NEBTab)
        self.btn_close_all.setObjectName(_fromUtf8("btn_close_all"))
        self.gridLayout_2.addWidget(self.btn_close_all, 1, 1, 1, 1)
        self.btn_rates = QtGui.QPushButton(self.NEBTab)
        self.btn_rates.setObjectName(_fromUtf8("btn_rates"))
        self.gridLayout_2.addWidget(self.btn_rates, 2, 1, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout_5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.tabWidget.addTab(self.NEBTab, _fromUtf8(""))
        self.TSTab = QtGui.QWidget()
        self.TSTab.setObjectName(_fromUtf8("TSTab"))
        self.gridLayout_8 = QtGui.QGridLayout(self.TSTab)
        self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
        self.oglTS = Show3DWithSlider(self.TSTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.oglTS.sizePolicy().hasHeightForWidth()
        )
        self.oglTS.setSizePolicy(sizePolicy)
        self.oglTS.setObjectName(_fromUtf8("oglTS"))
        self.gridLayout_8.addWidget(self.oglTS, 0, 0, 1, 1)
        self.verticalLayout_9 = QtGui.QVBoxLayout()
        self.verticalLayout_9.setObjectName(_fromUtf8("verticalLayout_9"))
        self.list_TS = QtGui.QTableView(self.TSTab)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.list_TS.sizePolicy().hasHeightForWidth()
        )
        self.list_TS.setSizePolicy(sizePolicy)
        self.list_TS.setMaximumSize(QtCore.QSize(300, 16777215))
        self.list_TS.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.list_TS.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.list_TS.setSortingEnabled(True)
        self.list_TS.setObjectName(_fromUtf8("list_TS"))
        self.list_TS.horizontalHeader().setCascadingSectionResizes(True)
        self.list_TS.horizontalHeader().setDefaultSectionSize(80)
        self.list_TS.horizontalHeader().setMinimumSectionSize(20)
        self.list_TS.verticalHeader().setVisible(False)
        self.verticalLayout_9.addWidget(self.list_TS)
        self.pushNormalmodesTS = QtGui.QPushButton(self.TSTab)
        self.pushNormalmodesTS.setObjectName(_fromUtf8("pushNormalmodesTS"))
        self.verticalLayout_9.addWidget(self.pushNormalmodesTS)
        self.gridLayout_8.addLayout(self.verticalLayout_9, 0, 2, 1, 1)
        self.tabWidget.addTab(self.TSTab, _fromUtf8(""))
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 839, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuSimulation = QtGui.QMenu(self.menubar)
        self.menuSimulation.setObjectName(_fromUtf8("menuSimulation"))
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName(_fromUtf8("menuHelp"))
        self.menuActions = QtGui.QMenu(self.menubar)
        self.menuActions.setObjectName(_fromUtf8("menuActions"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtGui.QAction(MainWindow)
        self.actionNew.setObjectName(_fromUtf8("actionNew"))
        self.actionClear = QtGui.QAction(MainWindow)
        self.actionClear.setObjectName(_fromUtf8("actionClear"))
        self.action_db_connect = QtGui.QAction(MainWindow)
        self.action_db_connect.setObjectName(_fromUtf8("action_db_connect"))
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName(_fromUtf8("actionAbout"))
        self.action_delete_minimum = QtGui.QAction(MainWindow)
        self.action_delete_minimum.setObjectName(
            _fromUtf8("action_delete_minimum")
        )
        self.action_edit_params = QtGui.QAction(MainWindow)
        self.action_edit_params.setObjectName(_fromUtf8("action_edit_params"))
        self.action_merge_minima = QtGui.QAction(MainWindow)
        self.action_merge_minima.setObjectName(_fromUtf8("action_merge_minima"))
        self.action_compute_thermodynamic_info = QtGui.QAction(MainWindow)
        self.action_compute_thermodynamic_info.setObjectName(
            _fromUtf8("action_compute_thermodynamic_info")
        )
        self.menuSimulation.addAction(self.action_db_connect)
        self.menuHelp.addAction(self.actionAbout)
        self.menuActions.addAction(self.action_delete_minimum)
        self.menuActions.addAction(self.action_merge_minima)
        self.menuActions.addAction(self.action_edit_params)
        self.menuActions.addAction(self.action_compute_thermodynamic_info)
        self.menubar.addAction(self.menuSimulation.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuActions.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate(
                "MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btn_stop_basinhopping.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Stop all basinhopping processes</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_stop_basinhopping.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Stop basin-hopping",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.pushNormalmodesMin.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>launch a tool to explore the normal modees of the current structure</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.pushNormalmodesMin.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Normalmodes",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.pushTakestepExplorer.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>launch a tool to look at the basinhopping steps in detail</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.pushTakestepExplorer.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Takestep explorer",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_heat_capacity.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>launch a tool to compute the heat capacity in the harmonic superposition approximation</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_heat_capacity.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "heat capacity",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_start_basinhopping.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Start a basinhopping run</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_start_basinhopping.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Start basin-hopping",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.label_bh_nproc.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>This lists the number of basinhopping processes currently running in parallel</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.label_bh_nproc.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "0 B.H. processes",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.lineEdit_bh_nsteps.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Set the number of basinhopping iterations</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.lineEdit_bh_nsteps.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "# B.H. steps",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.BHTab),
            QtGui.QApplication.translate(
                "MainWindow",
                "Basin Hopping",
                None,
                QtGui.QApplication.UnicodeUTF8,
            ),
        )
        self.btnAlign.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Find best alignment between two structures</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnAlign.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Align", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btnReconnect.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Start a fresh double ended connect run</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnReconnect.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Reconnect", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btnNEB.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Start an NEB run (no alignment is done)</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnNEB.setText(
            QtGui.QApplication.translate(
                "MainWindow", "NEB", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btnConnect.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Start a double ended connect run</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnConnect.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Connect", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btnShowGraph.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Show the graph of minima and transition states</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnShowGraph.setText(
            QtGui.QApplication.translate(
                "MainWindow", "Show graph", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.btnDisconnectivity_graph.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Show the disconnectivity graph</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btnDisconnectivity_graph.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Disconnectivity graph",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_connect_in_optim.setToolTip(
            QtGui.QApplication.translate(
                "MainWindow",
                "<html><head/><body><p>Spawn an external OPTIM job</p></body></html>",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_connect_in_optim.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Connect in OPTIM",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_connect_all.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Connect All",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_close_all.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Close windows",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.btn_rates.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Compute rates",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.NEBTab),
            QtGui.QApplication.translate(
                "MainWindow", "Connect", None, QtGui.QApplication.UnicodeUTF8
            ),
        )
        self.pushNormalmodesTS.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Normalmodes",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.TSTab),
            QtGui.QApplication.translate(
                "MainWindow",
                "Transition States",
                None,
                QtGui.QApplication.UnicodeUTF8,
            ),
        )
        self.menuSimulation.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Database", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuHelp.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.menuActions.setTitle(
            QtGui.QApplication.translate(
                "MainWindow", "Actions", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionNew.setText(
            QtGui.QApplication.translate(
                "MainWindow", "New", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.actionClear.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Clear Database",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.action_db_connect.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Connect to Database",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.actionAbout.setText(
            QtGui.QApplication.translate(
                "MainWindow", "About", None, QtGui.QApplication.UnicodeUTF8
            )
        )
        self.action_delete_minimum.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Delete Minimum",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.action_edit_params.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Edit default parameters",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.action_merge_minima.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Merge Minima",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )
        self.action_compute_thermodynamic_info.setText(
            QtGui.QApplication.translate(
                "MainWindow",
                "Compute thermodynamic info",
                None,
                QtGui.QApplication.UnicodeUTF8,
            )
        )


from .show3d_with_slider import Show3DWithSlider
from pele.gui.show3d import Show3D
