import vedo
from itertools import cycle

vedo.settings.enableDefaultKeyboardCallbacks = False
vedo.settings.enableDefaultMouseCallbacks = False

class Viewer:
    def __init__(self, *args, **kwargs):
        self.plotter = vedo.Plotter(*args, **kwargs) # setup the Plotter object
        self.keyevt = self.plotter.addCallback('KeyPressed', self.handle_key)
        self.angle = 0
        self.scaling = 2
        self.posY = cycle([-200, 368, 932])
        self.d_theta = 36

    def show(self, *args, **kwargs):
        plt = self.plotter.show(*args, **kwargs)
        return plt

    def handle_key(self, evt):
        if evt.keyPressed in ['Right']:
            self.angle += self.d_theta
            self.angle = 0 if self.angle == 360 else self.angle
            mesh.origin(0,0,0).rotateY(self.d_theta).show(resetcam=False)

        if evt.keyPressed in ['r']:
            mesh.origin(0,0,0).scale(self.scaling, True).y(self.scaling*next(self.posY)).show(resetcam=False)
        
        if evt.keyPressed in ['s']:
            angle = self.angle
            self.angle = 0 
            mesh.origin(0,0,0).y(0).scale(1, True).rotateY(360-angle).show(resetcam=False)

viewer = Viewer()
mesh = viewer.plotter.load('mesh/brenda2.ply')
viewer.show(mesh, viewup='y').close()