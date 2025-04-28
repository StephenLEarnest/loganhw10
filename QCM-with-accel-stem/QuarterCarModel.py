#region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height/2
        self.left = self.x - self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()

    def boundingRect(self):
        return self.transformation.mapRect(self.rect)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen:
            painter.setPen(self.pen)
        if self.brush:
            painter.setBrush(self.brush)
        self.top = -self.height/2
        self.left = -self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
#endregion

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.mass = mass
        self.transformation = qtg.QTransform()
        self.massBlock = MassBlock(CenterX, CenterY, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush)

    def boundingRect(self):
        return self.transformation.mapRect(qtc.QRectF(self.x-self.radius, self.y-self.radius, 2*self.radius, 2*self.radius))

    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen:
            painter.setPen(self.pen)
        if self.brush:
            painter.setBrush(self.brush)
        rect = qtc.QRectF(-self.radius, -self.radius, 2*self.radius, 2*self.radius)
        painter.drawEllipse(rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)

#endregion

#region MVC classes
class CarModel():
    def __init__(self):
        self.results = []
        self.tmax = 3.0
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)
        self.yangdeg = 45.0
        self.results = None
        self.m1 = 450
        self.m2 = 20
        self.c1 = 4500
        self.k1 = 15000
        self.k2 = 90000
        self.v = 120
        g = 9.81
        self.mink1 = (self.m1 * g) / (0.1524)
        self.maxk1 = (self.m1 * g) / (0.0762)
        self.mink2 = (self.m2 * g) / (0.0381)
        self.maxk2 = (self.m2 * g) / (0.01905)
        self.accel = None
        self.accelMax = 0.0
        self.accelLim = 2.0
        self.SSE = 0.0

class CarView():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, self.le_tmax, self.chk_IncludeAccel = self.input_widgets
        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)
        self.ax = self.figure.add_subplot()
        self.ax1 = self.ax.twinx()
        self.buildScene()

    def buildScene(self):
        self.scene = qtw.QGraphicsScene()
        self.scene.setSceneRect(-200, -200, 400, 400)
        self.gv_Schematic.setScene(self.scene)
        pen = qtg.QPen(qtg.QColor("orange"))
        brush = qtg.QBrush(qtg.QColor(255, 165, 0, 100))
        massBrush = qtg.QBrush(qtg.QColor(200, 200, 200, 128))
        self.wheel = Wheel(0, 50, 50, pen=pen, wheelBrush=brush, massBrush=massBrush)
        self.car = MassBlock(0, -70, 100, 30, pen=pen, brush=massBrush)
        self.wheel.addToScene(self.scene)
        self.scene.addItem(self.car)

    def updateView(self, model=None):
        self.le_m1.setText(f"{model.m1:.2f}")
        self.le_k1.setText(f"{model.k1:.2f}")
        self.le_c1.setText(f"{model.c1:.2f}")
        self.le_m2.setText(f"{model.m2:.2f}")
        self.le_k2.setText(f"{model.k2:.2f}")
        self.le_ang.setText(f"{model.yangdeg:.2f}")
        self.le_tmax.setText(f"{model.tmax:.2f}")
        self.lbl_MaxMinInfo.setText(f"k1_min={model.mink1:.2f}, k1_max={model.maxk1:.2f}\n"
                                    f"k2_min={model.mink2:.2f}, k2_max={model.maxk2:.2f}\nSSE={model.SSE:.2f}")
        self.doPlot(model)

    def doPlot(self, model):
        if model.results is None:
            return
        ax, ax1 = self.ax, self.ax1
        ax.clear()
        ax1.clear()
        t = model.t
        ycar = model.results[:,0]
        ywheel = model.results[:,2]
        accel = model.accel

        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        if self.chk_ShowAccel.isChecked():
            ax1.plot(t, accel, 'g-', label='Acceleration')

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vertical Position (m)")
        ax1.set_ylabel("Acceleration (g)")
        ax.legend()
        self.canvas.draw()

class CarController():
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, self.le_tmax, self.chk_IncludeAccel = self.input_widgets
        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets
        self.model = CarModel()
        self.view = CarView(args)

    def ode_system(self, X, t):
        x1, x1dot, x2, x2dot = X
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag
        m1, m2, k1, k2, c1 = self.model.m1, self.model.m2, self.model.k1, self.model.k2, self.model.c1
        x1ddot = (-k1*(x1-x2) - c1*(x1dot-x2dot)) / m1
        x2ddot = (k1*(x1-x2) + c1*(x1dot-x2dot) - k2*(x2-y)) / m2
        return [x1dot, x1ddot, x2dot, x2ddot]

    def doCalc(self, doPlot=True, doAccel=True):
        v = 1000 * self.model.v / 3600
        self.model.angrad = math.radians(self.model.yangdeg)
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)
        self.model.t = np.linspace(0, self.model.tmax, 2000)
        ic = [0, 0, 0, 0]
        self.model.results = odeint(self.ode_system, ic, self.model.t)
        if doAccel:
            self.calcAccel()
        if doPlot:
            self.view.doPlot(self.model)

    def calcAccel(self):
        vel = self.model.results[:,1]
        t = self.model.t
        self.model.accel = np.gradient(vel, t) / 9.81
        self.model.accelMax = np.max(self.model.accel)

    def SSE(self, vals, optimizing=True):
        k1, c1, k2 = vals
        self.model.k1 = k1
        self.model.c1 = c1
        self.model.k2 = k2
        self.doCalc(doPlot=False)
        SSE = np.sum((self.model.results[:,0] - np.interp(self.model.t, [0, self.model.t[-1]], [0, self.model.ymag]))**2)
        if optimizing:
            if (k1<self.model.mink1 or k1>self.model.maxk1 or
                k2<self.model.mink2 or k2>self.model.maxk2 or
                c1<10 or (self.model.accelMax>self.model.accelLim and self.chk_IncludeAccel.isChecked())):
                SSE += 1000
        self.model.SSE = SSE
        return SSE

    def calculate(self, doCalc=True):
        self.model.m1 = float(self.le_m1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.k2 = float(self.le_k2.text())
        self.model.v = float(self.le_v.text())
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())
        if doCalc:
            self.doCalc()
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def OptimizeSuspension(self):
        self.calculate(doCalc=False)
        x0 = np.array([self.model.k1, self.model.c1, self.model.k2])
        answer = minimize(self.SSE, x0, method='Nelder-Mead')
        self.model.k1, self.model.c1, self.model.k2 = answer.x
        self.doCalc()
        self.view.updateView(self.model)

    def doPlot(self):
        self.view.doPlot(self.model)
#endregion

#endregion

def main():
    app = qtw.QApplication([])
    mw = CarController()
    mw.doCalc()
    app.exec()

if __name__ == '__main__':
    main()
