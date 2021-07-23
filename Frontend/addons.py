import numpy as np
import multiprocessing as mp

from loading import Loading
from sklearn.cluster import KMeans
from vedo import Mesh, show, Plane, Plotter, interactive, io

def compareMesh(obj1, obj2):
    obj1 = Mesh(obj1).origin(0,0,0) 
    obj2 = Mesh(obj2).origin(0,0,0) 
    obj1.distanceToMesh(obj2, signed=True)
    show(obj1, interactive=False, new=True).export(".temp/dist2mesh.npy").close()

class Slicer:
    def __init__(self, obj:str, plotter:Plotter=None, n=6, levels=[3,3.5,4.3,5.1]):
        self.obj = Mesh(obj)
        self.min, self.max = extremes(self.obj.points()[:,1])
        self.n = n
        self.ratio = (self.max - self.min)/self.n
        self.plotter = plotter
        self.levels = levels
        self.currentLevel = 0
        self.previousLevel = None
    
    def foot(self):
        footPos = self.obj.centerOfMass()
        footPos[1] = self.min
        return footPos
    
    def getSlice(self):
        self.pl = Plane(self.foot() + [0, self.levels[self.currentLevel] * self.ratio, 0],
                   normal = [0,1,0], sx = 1E3, alpha = 1).c('green').bc('green').lw(4).lc('green')
        
        intersection = self.obj.intersectWith(self.pl).rotateX(90)
        sliceArray = intersection.points()
        sliceArray[:,-1] = 0
        sliceCentroid = intersection.centerOfMass()
        sliceCentroid[-1] = 0
        
        self.sliceArray = sliceArray
        self.sliceCentroid = sliceCentroid

    def prune(self):
        norm = np.linalg.norm(self.sliceArray-self.sliceCentroid, axis=1)
        if 3.5 <= self.levels[self.currentLevel] <= 4.5:
            kmeans = KMeans(n_clusters=2).fit(norm.reshape(-1,1))
            labels = kmeans.labels_
            labelTrue = kmeans.predict(np.min(norm).reshape(-1,1))
            prunedArray = self.sliceArray[labelTrue == labels]
            x, y, z = prunedArray[:,0], prunedArray[:,1], prunedArray[:,2]
        else:
            x, y, z = self.sliceArray[:,0], self.sliceArray[:,1], self.sliceArray[:,2]
        x, y = sortValues(x, y)
        return np.c_[x, y, z]
        
    def nextSlice(self):
        self.currentLevel = min(self.currentLevel + 1, len(self.levels) - 1)
        if self.currentLevel != self.previousLevel:
            self.previousLevel = self.currentLevel
            self.show()
        
    def prevSlice(self):
        self.currentLevel = max(self.currentLevel - 1, 0)
        if self.currentLevel != self.previousLevel:
            self.previousLevel = self.currentLevel
            self.show()
    
    def show(self):
        self.getSlice()
        array = self.prune()
        verts = np.c_[array.T, self.sliceCentroid].T[::-1]
        faces = [(0,i,j) if j!=len(verts) else (0,i,1) for i, j in zip(range(1,len(verts)+1),range(2,len(verts)+1))]
        mesh = Mesh([verts, faces]).clone2D(pos=[0.75, 0.5], coordsys=3, c='tomato', alpha=1, scale=.002)
        perimeter = round(arcLength(array[:,0], array[:,1])/10, 2)
        self.plotter.clear(at=0)
        self.plotter.show(self.obj, self.pl, mesh, f'Perímetro: {perimeter} cm', at=0, interactive=False)
        
def extremes(x):
        return np.min(x), np.max(x)
    
def sortValues(x, y):
    x0 = np.mean(x)
    y0 = np.mean(y)
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    angles = np.where((y-y0) > 0, np.arccos((x-x0)/r), 2*np.pi-np.arccos((x-x0)/r))
    mask = np.argsort(angles)
    return x[mask], y[mask]

def arcLength(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
    return arc
    
def colorMesh(obj, y1, y2):
    obj = Mesh(obj).origin(0,0,0)
    ids = obj.findCellsWithin(ybounds=(y1, y2))
    cols = [[255, 99, 71] if i in ids else [177, 177, 177] for i in range(obj.NCells())]  
    return cols

if __name__ == '__main__':
    plotter = Plotter()
    slice = Slicer(obj='mesh/Brenda2.ply')
    slice.show()