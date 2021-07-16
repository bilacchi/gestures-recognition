import json
import gc
import numpy as np
import pickle as pk

import multiprocessing as mp
from datetime import datetime
from itertools import cycle
from vedo import Mesh, Plotter, CornerAnnotation, Plane, interactive, settings

from Loading import Loading

gc.enable()

settings.enableDefaultMouseCallbacks = False
settings.enableDefaultKeyboardCallbacks = False
settings.allowInteraction = False

aspect = 16 / 9  # width / height
height = 700
width = np.ceil(height*aspect)

custom_shape = [dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg='w', bg2='w' ),
                dict(bottomleft=(0.0,0.0), topright=(0.20,0.30), bg='w', bg2='w')]

def compare(obj1, obj2):
    obj1 = Mesh(obj1).normalize().origin(0,0,0)
    obj2 = Mesh(obj2).normalize().origin(0,0,0)
    obj1.distanceToMesh(obj2, signed=True)
    dist2mesh = obj1.getPointArray('Distance').tolist()
    
    with open('.temp/dist2mesh.pkl', 'wb') as fpk:        
        pk.dump(dist2mesh, fpk)
        
def slice(obj, y1, y2):
    obj = Mesh(obj).normalize().origin(0,0,0)
    ids = obj.findCellsWithin(ybounds=(y1, y2))
    cols = [[255, 99, 71] if i in ids else [177, 177, 177] for i in range(obj.NCells())]  
    
    with open('.temp/sliceMesh.pkl', 'wb') as fpk:
        pk.dump((id, cols), fpk)

class Viewer:
    def __init__(self, timeline, size=(1200, 700)):
        self.windowSize = size
        self.plotter = Plotter(shape=custom_shape, size=self.windowSize)
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
        self.mesh = Mesh(mesh).normalize()

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        txt2d = CornerAnnotation().font('Arial').text(txt)
        self.plotter.show(self.mesh, txt2d, title='Brenda Plot', viewup='y', at=0, interactorStyle=12)

    def changeDate(self):
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = Mesh(mesh).normalize()

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        txt2d = CornerAnnotation().font('Arial').text(txt)
        
        self.plotter.show(self.mesh, txt2d, title='Brenda Plot', viewup='z', at=0, interactorStyle=12)

    def clearMiniWindow(self):
        self.plotter.clear(at=1)
        
    def clearMainWindow(self):
        self.plotter.clear(at=0)
         
    def meshOrigin(self):
        self.mesh.origin(0, 0, 0).rotateY(360-self.angle)
        self.angle = 0
        
    def slicePlane(self):
        _ , mesh = self.timeline[self.meshSlice[0]][self.meshSlice[1]]
        y1, y2 = self.zslice[self.meshSliceIndex]
        
        Loading(job=mp.Process(target=slice, args=(mesh, y1, y2)),
                plotter=self.plotter).run()
        
        ## Add load pickle
        with open('.temp/sliceMesh.pkl', 'rb') as fpk:
            _, cols = pk.load(fpk)
        
        self.mesh.cellIndividualColors(cols)
        self.meshOrigin()

        p1 = Plane(normal=(0,1,0), sx=2, sy=2).y(y1).c('gray',0.5)
        p2 = p1.clone().y(y2)
        
        self.clearMainWindow()
        self.plotter.show(self.mesh, p1, p2, at=0, resetcam=True)
        
    def handle_key(self, evt):
        if evt.keyPressed in ['r']:
            if self.meshSlice is not None:
                previousIndex = self.meshSliceIndex
                self.meshSliceIndex = min(self.meshSliceIndex+1, len(self.zslice)-1)
                if self.meshSliceIndex != previousIndex:
                    self.slicePlane()
                
            elif self.meshSlice is None:
                self.angle += self.d_theta
                self.angle = 0 if self.angle == 360 else self.angle
                self.mesh.origin(0, 0, 0).rotateY(self.d_theta)
                self.plotter.render()
                
            else:
                NotImplementedError

        if evt.keyPressed in ['l']:
            if self.meshSlice is not None:
                previousIndex = self.meshSliceIndex
                self.meshSliceIndex = max(self.meshSliceIndex-1, 0)
                if self.meshSliceIndex != previousIndex:
                    self.slicePlane()
                
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
            self.clearMiniWindow()
            self.clearMainWindow()
            
            if self.meshSlice is None and self.meshComp is None:
                self.meshSliceIndex = 0
                self.meshSlice = (self.current_exam, self.index)
                self.slicePlane()

            elif self.meshSlice is not None:
                self.meshSlice = None
                self.changeDate()
            
        if evt.keyPressed in ['s']:
            if self.meshSlice is None:
                if self.meshComp is None:
                    self.meshComp = (self.current_exam, self.index)
                    _ , mesh = self.timeline[self.meshComp[0]][self.meshComp[1]]
                    self.plotter.add(self.mesh.clone(), at=1, resetcam=False)
                    
                elif self.meshComp == 'BLOCK':
                    self.meshComp == None
                    self.changeDate()
                    
                elif self.meshComp is not None:
                    _ , mesh1 = self.timeline[self.meshComp[0]][self.meshComp[1]]
                    _ , mesh2 = self.timeline[self.current_exam][self.index]
                    
                    self.clearMiniWindow()
                    Loading(job=mp.Process(target=compare, args=(mesh1, mesh2)),
                            plotter=self.plotter).run()
                    
                    meshCompResult = Mesh('.temp/temp.ply').normalize().origin(0,0,0).cmap('jet')

                    with open('.temp/dist2mesh.pkl', 'rb') as fpk:
                        dist2mesh = pk.load(fpk)
                        
                    meshCompResult._mapper.SetScalarRange(min(dist2mesh), max(dist2mesh))
                    meshCompResult._mapper.ScalarVisibilityOn()
                    meshCompResult.addScalarBar('Distância')
                    
                    self.mesh = meshCompResult
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