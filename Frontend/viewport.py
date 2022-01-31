import gc
import os
import json
import numpy as np

import multiprocessing as mp
from datetime import datetime
from itertools import cycle
from vedo import Mesh, Plotter, CornerAnnotation, interactive, settings, io
from addons import compareMesh3D, Slicer

from loading import Loading

try: os.mkdir('.temp')
except: pass

gc.enable()

settings.enableDefaultMouseCallbacks = False
settings.enableDefaultKeyboardCallbacks = False
settings.allowInteraction = False

aspect = 16 / 9  # width / height
height = 700
width = np.ceil(height*aspect)

custom_shape = [dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg='w', bg2='w' ),
                dict(bottomleft=(0.0,0.0), topright=(0.20,0.30), bg='w', bg2='w')]
        

class Viewer:
    def __init__(self, timeline, size=(1200, 700)):
        self.windowSize = size
        self.plotter = Plotter(shape=custom_shape, size=self.windowSize)
        self.keyevt = self.plotter.addCallback('KeyPressed', self.handle_key)
        self.angle = 0
        self.d_theta = 36
        self.meshComp = None
        self.meshSlice = None
        self.camera = {}
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
        self.mesh = Mesh(mesh) 

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        txt2d = CornerAnnotation().font('Arial').text(txt)
        self.plotter.show(self.mesh, txt2d, title='Brenda Plot', viewup='y', at=0, interactorStyle=12)
        self.camera['Position'] = self.plotter.camera.GetPosition()
        self.camera['FocalPoint'] = self.plotter.camera.GetFocalPoint()
        self.camera['ParallelScale'] = self.plotter.camera.GetParallelScale()
        self.camera['ViewUp'] = self.plotter.camera.GetViewUp()

    def changeDate(self):
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = Mesh(mesh).origin(0,0,0)
        
        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        txt2d = CornerAnnotation().font('Arial').text(txt)
        
        self.clearMainWindow()
        self.plotter.show(self.mesh, txt2d, resetcam=False, title='Brenda Plot', viewup='z', at=0, interactorStyle=12)

    def clearMiniWindow(self):
        self.plotter.clear(at=1)
        
    def clearMainWindow(self):
        self.plotter.clear(at=0)
        self.plotter.camera.SetPosition(self.camera['Position'])
        self.plotter.camera.SetFocalPoint(self.camera['FocalPoint'])
        self.plotter.camera.SetParallelScale(self.camera['ParallelScale'])
        self.plotter.camera.SetViewUp(self.camera['ViewUp'])
         
    def meshOrigin(self, mesh=None):
        if mesh is None:
            self.mesh.origin(0, 0, 0).rotateY(360-self.angle)
            self.angle = 0
        
        elif isinstance(mesh, Mesh):
            return mesh.origin(0, 0, 0).rotateY(360-self.angle)
        
        else:
            NotImplementedError
        
    def handle_key(self, evt):
        if evt.keyPressed in ['r']:
            if self.meshSlice is not None:
                self.slicer.nextSlice()
                
            elif self.meshSlice is None:
                self.angle += self.d_theta
                self.angle = 0 if self.angle == 360 else self.angle
                self.mesh.origin(0, 0, 0).rotateY(self.d_theta)
                self.plotter.render()
                
            else:
                NotImplementedError

        if evt.keyPressed in ['l']:
            if self.meshSlice is not None:
                self.slicer.prevSlice()
                
            elif self.meshSlice is None:
                self.angle -= self.d_theta
                self.angle = 360 if self.angle == 0 else self.angle
                self.mesh.origin(0, 0, 0).rotateY(-self.d_theta)
                self.plotter.render()
                
            else:
                NotImplementedError
            
        if evt.keyPressed in ['Right']:
            if self.meshSlice is None:
                self.meshComp = None if self.meshComp == 'BLOCK' else self.meshComp
                self.index = max(self.index-1, 0)
                self.clearMainWindow()
                self.meshOrigin()
                self.changeDate()

        if evt.keyPressed in ['Left']:
            if self.meshSlice is None:
                self.meshComp = None if self.meshComp == 'BLOCK' else self.meshComp
                self.index = min(self.index+1, len(self.timeline[self.current_exam])-1)
                self.clearMainWindow()
                self.meshOrigin()
                self.changeDate()

        if evt.keyPressed in ['Up']:
            if self.meshSlice is None and self.meshComp is None:
                self.meshSlice = 'ACTIVE'
                self.clearMainWindow()
                _ , mesh = self.timeline[self.current_exam][self.index]
                self.slicer = Slicer(mesh, self.plotter)
                self.slicer.show()

            elif self.meshSlice is not None:
                self.meshSlice = None
                self.changeDate()
            
        if evt.keyPressed in ['s']:
            if self.meshSlice is None:
                if self.meshComp is None:
                    self.meshComp = (self.current_exam, self.index)
                    self.plotter.add(self.meshOrigin(self.mesh.clone()), at=1, resetcam=False)
                    
                elif self.meshComp == 'BLOCK':
                    self.meshComp = None
                    self.changeDate()
                    
                elif self.meshComp is not None:
                    _ , mesh1 = self.timeline[self.meshComp[0]][self.meshComp[1]]
                    _ , mesh2 = self.timeline[self.current_exam][self.index]
                    
                    self.clearMiniWindow()
                    Loading(plotter=self.plotter).run(job=mp.Process(target=compareMesh3D, args=(mesh1, mesh2)))
                    
                    scals = np.load('.temp/dist2mesh.npy')
                    self.mesh.origin(0,0,0).cmap('Spectral_r', scals).addScalarBar()
                
                    self.meshComp = 'BLOCK'
                    self.plotter.show(self.mesh, at=0, resetcam=True)
                
with open('timeline.json', 'r') as fjson:
    timeline = json.load(fjson)

def main():
    global timeline
    Viewer(timeline, size=(width, height)).initialize()
    interactive().close()

if __name__ == '__main__':
    main()