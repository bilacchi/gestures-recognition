import vedo
import json
import numpy as np

from time import sleep
from datetime import datetime
from itertools import cycle

vedo.settings.enableDefaultKeyboardCallbacks = False
vedo.settings.enableDefaultMouseCallbacks = False

aspect = 16 / 9  # width / height
height = 700
width = np.ceil(height*aspect)

class Viewer:
    def __init__(self, timeline, size=(1200, 700)):
        self.windowSize = size
        custom_shape = [
                dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg='w', bg2='w' ),# ren0
                dict(bottomleft=(0.0,0.0), topright=(0.20,0.30), bg='w', bg2='w')# ren1
            ]
        self.plotter = vedo.Plotter(shape=custom_shape, size=self.windowSize)
        self.keyevt = self.plotter.addCallback('KeyPressed', self.handle_key)
        self.timerevt = self.plotter.addCallback('timer', self.handle_timer)
        
        self.angle = 0
        self.d_theta = 36
        self.meshComp = None
        self.exams = cycle(list(timeline.keys()))
        self.timeline = {}
        for key in timeline.keys():
            self.timeline[key] = sorted(zip(timeline[key]['date'], timeline[key]['file']), reverse=True)

    def initialize(self):
        self.current_exam = next(self.exams)
        self.index = 0

        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = vedo.Mesh(mesh).normalize()

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d = vedo.CornerAnnotation().font('Arial').text(txt)
        self.plotList = [self.mesh, self.txt2d]
        self.plotter.show(*self.plotList, title='Brenda Plot', viewup='y', at=0)

    def changeExam(self):
        self.timeline[self.current_exam][self.index]
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = vedo.Mesh(mesh).normalize()

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d.text(txt)
        
        self.plotList = [self.mesh, self.txt2d]
        vedo.clear(at=0)
        self.plotter.show(*self.plotList, title='Brenda Plot', viewup='z', at=0)

    def changeDate(self):
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = vedo.Mesh(mesh).normalize()

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d.text(txt)
        self.plotList = [self.mesh, self.txt2d]
        vedo.clear(at=0)
        self.plotter.show(*self.plotList, title='Brenda Plot', viewup='z', at=0)

    def miniWindow(self):
        if self.meshComp is not None:
            self.meshComp = None
            vedo.clear(at=1)
            self.plotter.render()
    
    def handle_timer(self, evt):
        pass
    
    def handle_key(self, evt):
        if evt.keyPressed in ['r']:
            self.angle += self.d_theta
            self.angle = 0 if self.angle == 360 else self.angle
            if isinstance(self.mesh, list):
                for mesh in self.mesh:
                    mesh.origin(0, 0, 0).rotateY(self.d_theta)
            else:
                self.mesh.origin(0, 0, 0).rotateY(self.d_theta)
            self.plotter.render()

        if evt.keyPressed in ['l']:
            self.angle -= self.d_theta
            self.angle = 360 if self.angle == 0 else self.angle
            if isinstance(self.mesh, list):
                for mesh in self.mesh:
                    mesh.origin(0, 0, 0).rotateY(-self.d_theta)
            else:
                self.mesh.origin(0, 0, 0).rotateY(-self.d_theta)
            self.plotter.render()
            
        if evt.keyPressed in ['Right']:
            self.index = max(self.index-1, 0)
            self.changeDate()

        if evt.keyPressed in ['Left']:
            self.index = min(self.index+1, len(self.timeline[self.current_exam])-1)
            self.changeDate()

        if evt.keyPressed in ['Up']:
            self.current_exam = next(self.exams)
            self.changeExam()
            self.miniWindow()
            
        if evt.keyPressed in ['s']:
            if self.meshComp is None:
                self.meshComp = (self.current_exam, self.index)
                _ , mesh = self.timeline[self.meshComp[0]][self.meshComp[1]]
                self.plotter.add(vedo.Mesh(mesh).normalize(), at=1)

            elif self.meshComp is not None and self.current_exam == 'Gordura':
                _ , mesh = self.timeline[self.meshComp[0]][self.meshComp[1]]
                mesh = vedo.Mesh(mesh).normalize()
                
                _ , mesh2 = self.timeline[self.current_exam][self.index]
                mesh2 = vedo.Mesh(mesh2).normalize()

                mesh.distanceToMesh(mesh2, signed=True)
                mesh.addScalarBar(title='Signed\nDistance')
                
                self.mesh = mesh
                self.miniWindow()
                self.plotter.show(mesh, at=0)
            
            elif self.meshComp is not None and self.current_exam == 'Postura':
                _ , mesh = self.timeline[self.meshComp[0]][self.meshComp[1]]
                mesh = vedo.Mesh(mesh).normalize()
                
                meshdec = mesh.clone().triangulate().decimate(N=200)

                sources = [[0.9, 0.0, 0.2]]  # this point moves
                targets = [[1.2, 0.0, 0.4]]  # to this.

                for pt in meshdec.points():
                    if pt[0] < 0.3:          # these pts don't move
                        sources.append(pt)   # source = target
                        targets.append(pt)   #

                warp = mesh.clone().thinPlateSpline(sources, targets)
                warp.c("blue",0.3).lineWidth(0)

                apts = vedo.Points(sources).c("red")
                
                self.mesh = [mesh, warp, apts]
                self.miniWindow()
                self.plotter.show(mesh, warp, apts, __doc__, viewup="y", at=0)
                
                

with open('timeline.json', 'r') as fjson:
    timeline = json.load(fjson)

def main():
    global timeline
    Viewer(timeline, size=(width, height)).initialize()
    vedo.interactive().close()

if __name__ == '__main__':
    main()