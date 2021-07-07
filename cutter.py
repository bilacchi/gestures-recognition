"""Drag the sphere to cut the mesh interactively
Use mouse buttons to zoom and pan"""
from vedo import *

s = Mesh('mesh/brenda2.ply')

plt = show(s, __doc__, bg='black', bg2='bb', interactive=False)
plt.addCutterTool(s) #modes= sphere, plane, box
