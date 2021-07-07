import vedo
import json
import numpy as np
from datetime import datetime
from itertools import cycle

vedo.settings.enableDefaultKeyboardCallbacks = False
vedo.settings.enableDefaultMouseCallbacks = False


class Viewer:
    def __init__(self, timeline, *args, **kwargs):
        # setup the Plotter object
        self.plotter = vedo.Plotter(*args, **kwargs)
        self.keyevt = self.plotter.addCallback('KeyPressed', self.handle_key)
        self.angle = 0
        self.scaling = 2
        self.d_theta = 36
        self.exams = cycle(list(timeline.keys()))
        self.timeline = {}
        for key in timeline.keys():
            self.timeline[key] = sorted(zip(timeline[key]['date'], timeline[key]['file']), reverse=True)
        
    def initialize(self):
        self.current_exam = next(self.exams)
        self.index = 0

        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = self.plotter.load(mesh)

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d = vedo.CornerAnnotation().font('Arial').text(txt)

        self.show(self.mesh, self.txt2d,
                  title='Brenda Plot', viewup='y').close()

    def changeExam(self):
        self.timeline[self.current_exam][self.index]
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = self.plotter.load(mesh)

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d.text(txt)

        self.show(self.mesh, self.txt2d,
                  title='Brenda Plot', viewup='z').close()

    def changeDate(self):
        date, mesh = self.timeline[self.current_exam][self.index]
        self.mesh = self.plotter.load(mesh)

        txt = f'{self.current_exam}\n'
        txt += datetime.strptime(str(date), '%y%m%d').strftime('%d %B, %Y')
        self.txt2d.text(txt)

        self.show(self.mesh, self.txt2d,
                  title='Brenda Plot', viewup='z').close()

    def show(self, *args, **kwargs):
        plt = self.plotter.show(*args, **kwargs)
        return plt

    def handle_key(self, evt):
        if evt.keyPressed in ['r']:
            self.angle += self.d_theta
            self.angle = 0 if self.angle == 360 else self.angle
            self.mesh.origin(0, 0, 0).rotateY(self.d_theta)
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

        if evt.keyPressed in ['s']:
            angle = self.angle
            self.angle = 0
            self.mesh.origin(0, 0, 0).y(0).scale(1, True).rotateY(360-angle)
            self.plotter.render()


with open('timeline.json', 'r') as fjson:
    timeline = json.load(fjson)


def main():
    global timeline
    Viewer(timeline).initialize()


if __name__ == '__main__':
    main()
