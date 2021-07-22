from vedo import shapes, show, interactive, Plotter
from numpy import pi, cos, sin
import multiprocessing as mp
import time

class Loading:
    def __init__(self, plotter, dt=-0.03, side=.3, r=1, ncubes=12, color='teal'):
        self.cubes = [shapes.Cube(side=side).x(0).y(0).z(0).c(color) for _ in range(ncubes)]
        self.dt = dt
        self.r = r
        self.plotter = plotter
        self.step = 2*pi/len(self.cubes)
        self.angle = 0
        self.initialize()

    def initialize(self):
        self.plotter.clear(at=0)
        for n, cube in enumerate(self.cubes):
            x = self.r * cos(n * self.step)
            y = self.r * sin(n * self.step)
            cube.x(x).y(y)
        self.plotter.show(*self.cubes, interactive=False, resetcam=True, zoom=1/3, at=0)
 
    def animate(self):
        self.angle += self.dt 
        for n, cube in enumerate(self.cubes):
            x = self.r * cos(self.angle + n*self.step)
            y = self.r * sin(self.angle + n*self.step)
            cube.x(x).y(y)

        self.plotter.show(*self.cubes, interactive=False, resetcam=False, at=0)
    
    def run(self, job):
        job.start()
        while job.is_alive():
            self.animate()


if __name__ == '__main__':
    plt = Plotter(size=(1280,720),
                    shape=[
                        dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg='w', bg2='w'),
                        dict(bottomleft=(0.0,0.0), topright=(0.20,0.30), bg='w', bg2='w')])
    
    plt.show(shapes.Cube(), at=0)
    plt.show(shapes.Cube(), at=1)

    anim = Loading(job=mp.Process(target=lambda: time.sleep(60)), plotter=plt)
    anim.run()
    interactive().close()