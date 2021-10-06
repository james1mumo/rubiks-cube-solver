import numpy as np

faces = ['y', 'w', 'g', 'b', 'r', 'o']

cube = [
    ['y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y'],
    ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
    ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'],
    ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
    ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'],
    ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
]


def perform_r_move():
    """affected faces are all except left face"""
    new_cube = []
    top_face = ''


