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
        
        self.angle = 0
        self.d_theta = 36
        self.meshComp = None
        self.meshSlice = None
        self.exams = cycle(list(timeline.keys()))
        vec = np.linspace(-2.7, 0.75, 6)
        self.zslice = np.round(list(zip(vec[:-1], vec[1:])), 2).tolist()
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
    
    def slicePlane(self):
        _ , mesh = self.timeline[self.current_exam][self.index]
        mesh = vedo.Mesh(mesh).normalize()
        
        y1, y2 = self.zslice[self.meshSlice]
        ids = mesh.findCellsWithin(ybounds=(y1, y2))

        cols = [[255, 99, 71] if i in ids else [177, 177, 177] for i in range(mesh.NCells())]  
        mesh.cellIndividualColors(cols)

        p1 = vedo.Plane(normal=(0,1,0), sx=2, sy=2).y(y1).c('gray',0.5)
        p2 = p1.clone().y(y2)
                
        self.mesh = mesh
        vedo.clear(at=0)
        self.plotter.show(mesh, p1, p2, viewup="y", at=0)
        
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
            if self.meshSlice is None:
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
                    self.plotter.show(mesh, warp, apts, viewup="y", at=0)
                
        if evt.keyPressed in ['f']:
            if self.meshSlice is None:
                self.meshSlice = 0
                self.slicePlane()

            elif self.meshSlice is not None:
                self.changeDate()
                self.meshSlice = None
            
        if evt.keyPressed in ['a']:
            if self.meshSlice is not None:
                self.meshSlice = max(self.meshSlice-1, 0)
                self.slicePlane()

        
        if evt.keyPressed in ['d']:
            if self.meshSlice is not None:
                self.meshSlice = min(self.meshSlice+1, len(self.zslice)-1)
                self.slicePlane()

with open('timeline.json', 'r') as fjson:
    timeline = json.load(fjson)

def main():
    global timeline
    Viewer(timeline, size=(width, height)).initialize()
    vedo.interactive().close()

if __name__ == '__main__':
    main()