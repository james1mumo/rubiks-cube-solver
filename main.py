"""
Solve a Rubiks Cube
- Take picture of rubiks cube
- Script solves the cube
- Shows you all moves till you get the result
"""

import cv2
import os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
from numpy import sin, cos
from matplotlib.patches import Rectangle, Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from PIL import Image
import numpy as np

faces = ['y', 'w', 'g', 'b', 'r', 'o']
# cube = [[], [], [], [], [], []]

cube = [
    ['g', 'r', 'y', 'b', 'y', 'y', 'o', 'o', 'b'],
    ['w', 'w', 'y', 'g', 'w', 'b', 'b', 'b', 'b'],
    ['o', 'y', 'y', 'w', 'g', 'r', 'o', 'r', 'r'],
    ['o', 'o', 'b', 'g', 'b', 'b', 'r', 'r', 'r'],
    ['r', 'w', 'w', 'w', 'r', 'o', 'w', 'o', 'w'],
    ['o', 'y', 'y', 'y', 'o', 'y', 'g', 'g', 'g'],
]

cube = [
    ['y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y'],
    ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
    ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'],
    ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
    ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'],
    ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
]

current_front_face = 4  # starts with red at front

up_face = 'y'
pics_directory = 'rubiks-cube-pics'
box_pixels = 80


def get_cube_pics():
    """
    Takes pictures of the cube faces
    and creates a model of the cube in memory

    :return:
    """

    window_name = 'Take Rubiks Cube Faces Pictures'

    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    count = 1

    print(f'[+]Press ESC to quit\n'
          f'[+]Press SPACE to take a picture')

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # FLip camera for easier cube positioning
        frame = cv2.flip(frame, 1)

        window_width = frame.shape[1]
        window_height = frame.shape[0]

        # Draw grid shapes on the camera window
        cv2.line(img=frame, pt1=(0, int(window_height / 2 - box_pixels / 2)),
                 pt2=(window_width, int(window_height / 2 - box_pixels / 2)), color=(255, 255, 255), thickness=1)
        cv2.line(img=frame, pt1=(0, int(window_height / 2 + box_pixels / 2)),
                 pt2=(window_width, int(window_height / 2 + box_pixels / 2)), color=(255, 255, 255), thickness=1)
        cv2.line(img=frame, pt1=(int(window_width / 2 - box_pixels / 2), 0),
                 pt2=(int(window_width / 2 - box_pixels / 2), window_height), color=(255, 255, 255), thickness=1)
        cv2.line(img=frame, pt1=(int(window_width / 2 + box_pixels / 2), 0),
                 pt2=(int(window_width / 2 + box_pixels / 2), window_height), color=(255, 255, 255), thickness=1)

        cv2.imshow(window_name, frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            if not os.path.isdir(pics_directory):
                os.mkdir(pics_directory)
            cv2.imwrite(f'{pics_directory}/{count}.jpg', frame)

            print(f"Took Picture {count}")

            if count == 6:
                break
            count += 1

    cam.release()
    cv2.destroyAllWindows()


def get_nearest_cube_color(rgb):
    colors = (
        (255, 255, 0, "yellow"),
        (255, 255, 255, "white"),
        (0, 128, 0, "green"),
        (0, 0, 255, "blue"),
        (255, 0, 0, "red"),
        (255, 165, 0, "orange")
    )
    return min(colors, key=lambda color: sum((s - q) ** 2 for s, q in zip(color, rgb)))[3]


def pics_to_cube():
    for i in range(6):
        img = cv2.imread(f'{pics_directory}/{i + 1}.jpg')
        half_img_height = img.shape[0] / 2
        half_img_width = img.shape[1] / 2

        face = [
            img[int(half_img_height - box_pixels), int(half_img_width - box_pixels)],
            img[int(half_img_height - box_pixels), int(half_img_width)],
            img[int(half_img_height - box_pixels), int(half_img_width + box_pixels)],

            img[int(half_img_height), int(half_img_width - box_pixels)],
            img[int(half_img_height), int(half_img_width)],
            img[int(half_img_height), int(half_img_width + box_pixels)],

            img[int(half_img_height + box_pixels), int(half_img_width - box_pixels)],
            img[int(half_img_height + box_pixels), int(half_img_width)],
            img[int(half_img_height + box_pixels), int(half_img_width + box_pixels)],
        ]

        b, g, r = img[int(half_img_width), int(half_img_height)]
        foo = get_nearest_cube_color((r, g, b))

        pass
        for box in face:
            foo = get_nearest_cube_color(box)
            # cv2.drawMarker(img, (int(half_img_width - box_pixels), int(half_img_height - box_pixels)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width - box_pixels), int(half_img_height)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width - box_pixels), int(half_img_height + box_pixels)), (255, 255, 255))
            #
            # cv2.drawMarker(img, (int(half_img_width), int(half_img_height - box_pixels)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width), int(half_img_height)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width), int(half_img_height + box_pixels)), (255, 255, 255))
            #
            # cv2.drawMarker(img, (int(half_img_width + box_pixels), int(half_img_height - box_pixels)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width + box_pixels), int(half_img_height)), (255, 255, 255))
            # cv2.drawMarker(img, (int(half_img_width + box_pixels), int(half_img_height + box_pixels)), (255, 255, 255))

            pass

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('f', img_gray)
        cv2.waitKey()
        print('')


def get_back_face(front_face):
    if (front_face % 2) == 0:
        return front_face + 1
    else:
        return front_face - 1


def get_right_face(front_face):
    if front_face == 2:
        return 5
    elif front_face == 3:
        return 4
    elif front_face == 4:
        return 2
    elif front_face == 5:
        return 3


def get_left_face(front_face):
    if front_face == 2:
        return 4
    elif front_face == 3:
        return 5
    elif front_face == 4:
        return 3
    elif front_face == 5:
        return 2


def set_front_face(front_face):
    global current_front_face
    if current_front_face == front_face:
        return

    # red[4] --> right of red is --> green[2] and so on
    # faces = r, g, o, b
    faces = [4, 2, 5, 3]

    rotations = faces.index(front_face) - faces.index(current_front_face)
    if rotations < 0:
        rotations += 4
    upper_face = np.array_split(cube[0], 3)
    bottom_face = np.array_split(cube[1], 3)

    cube[0] = list(np.concatenate(np.rot90(upper_face, 4 - rotations)))  # Upper face is rotated clockwise
    cube[1] = list(np.concatenate(np.rot90(bottom_face, rotations)))  # Bottom face is rotated anti-clockwise

    current_front_face = front_face


def move_right_face(front_face):
    old_cube = copy.deepcopy(cube)
    back_face = get_back_face(front_face)
    right_face = get_right_face(front_face)

    # Update top yellow face . front face comes to top yellow face
    cube[0][2] = old_cube[front_face][2]
    cube[0][5] = old_cube[front_face][5]
    cube[0][8] = old_cube[front_face][8]

    # Update front face . bottom white face comes to front face
    cube[front_face][2] = old_cube[1][2]
    cube[front_face][5] = old_cube[1][5]
    cube[front_face][8] = old_cube[1][8]

    # Update back face . top yellow face  comes to back face
    cube[back_face][0] = old_cube[0][8]
    cube[back_face][3] = old_cube[0][5]
    cube[back_face][6] = old_cube[0][2]

    # Update bottom face . back  face  comes to bottom face
    cube[1][2] = old_cube[back_face][6]
    cube[1][5] = old_cube[back_face][3]
    cube[1][8] = old_cube[back_face][0]

    # Update right face . right faces just rotates
    cube[right_face][0] = old_cube[right_face][6]
    cube[right_face][1] = old_cube[right_face][3]
    cube[right_face][2] = old_cube[right_face][0]

    cube[right_face][3] = old_cube[right_face][7]
    cube[right_face][4] = old_cube[right_face][4]
    cube[right_face][5] = old_cube[right_face][1]

    cube[right_face][6] = old_cube[right_face][8]
    cube[right_face][7] = old_cube[right_face][5]
    cube[right_face][8] = old_cube[right_face][2]


def move_upper_face(front_face):
    old_cube = copy.deepcopy(cube)
    back_face = get_back_face(front_face)
    right_face = get_right_face(front_face)
    left_face = get_left_face(front_face)
    rotating_face = 0

    cube[front_face][0] = old_cube[right_face][0]
    cube[front_face][1] = old_cube[right_face][1]
    cube[front_face][2] = old_cube[right_face][2]

    cube[left_face][0] = old_cube[front_face][0]
    cube[left_face][1] = old_cube[front_face][1]
    cube[left_face][2] = old_cube[front_face][2]

    cube[back_face][0] = old_cube[left_face][0]
    cube[back_face][1] = old_cube[left_face][1]
    cube[back_face][2] = old_cube[left_face][2]

    cube[right_face][0] = old_cube[back_face][0]
    cube[right_face][1] = old_cube[back_face][1]
    cube[right_face][2] = old_cube[back_face][2]

    # Rotating face
    cube[rotating_face][0] = old_cube[rotating_face][6]
    cube[rotating_face][1] = old_cube[rotating_face][3]
    cube[rotating_face][2] = old_cube[rotating_face][0]

    cube[rotating_face][3] = old_cube[rotating_face][7]
    cube[rotating_face][4] = old_cube[rotating_face][4]
    cube[rotating_face][5] = old_cube[rotating_face][1]

    cube[rotating_face][6] = old_cube[rotating_face][8]
    cube[rotating_face][7] = old_cube[rotating_face][5]
    cube[rotating_face][8] = old_cube[rotating_face][2]


def move_left_face(front_face):
    set_front_face(get_back_face(front_face))
    move_right_face(front_face)
    set_front_face(front_face)


def move_front_face(front_face):
    set_front_face(get_left_face(front_face))
    move_right_face(front_face)
    set_front_face(front_face)


def move_right_face_inverse(front_face):
    """simply move the cube 3 right moves and that's its inverse """
    move_right_face(front_face)
    move_right_face(front_face)
    move_right_face(front_face)


def move_upper_face_inverse(front_face):
    move_upper_face(front_face)
    move_upper_face(front_face)
    move_upper_face(front_face)

def move_left_face_inverse(front_face):
    move_right_face(front_face)
    move_right_face(front_face)
    move_right_face(front_face)

def move_front_face_inverse(front_face):
    move_front_face(front_face)
    move_front_face(front_face)
    move_front_face(front_face)

def perform_right_hand_rule(front_face):
    move_right_face(front_face)
    move_upper_face(front_face)
    move_right_face_inverse(front_face)
    move_upper_face_inverse(front_face)

def perform_left_hand_rule(front_face):
    move_left_face_inverse(front_face)
    move_upper_face_inverse(front_face)
    move_left_face(front_face)
    move_upper_face(front_face)

def append_images(images, direction='horizontal', bg_color=(255, 255, 255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction == 'horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1]) / 2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0]) / 2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def draw_cube(front_face):
    colors = [
        Image.new('RGB', (30, 30), color='yellow'),
        Image.new('RGB', (30, 30), color='white'),
        Image.new('RGB', (30, 30), color='green'),
        Image.new('RGB', (30, 30), color='blue'),
        Image.new('RGB', (30, 30), color='red'),
        Image.new('RGB', (30, 30), color='orange')
    ]
    face_images = []
    for face in cube:
        face_top_img = append_images(
            [colors[faces.index(face[0])], colors[faces.index(face[1])], colors[faces.index(face[2])]])
        face_middle_img = append_images(
            [colors[faces.index(face[3])], colors[faces.index(face[4])], colors[faces.index(face[5])]])
        face_bottom_img = append_images(
            [colors[faces.index(face[6])], colors[faces.index(face[7])], colors[faces.index(face[8])]])

        face_img = append_images([face_top_img, face_middle_img, face_bottom_img], direction='vertical')
        face_images.append(face_img)

    black_img = Image.new('RGB', (30, 30), color='black')
    padding_img = append_images([black_img, black_img, black_img])
    padding_img = append_images([padding_img, padding_img, padding_img], direction='vertical')

    vertical_cube_img = append_images([face_images[get_back_face(front_face)], face_images[0], face_images[front_face],
                                       face_images[1]], direction='vertical')
    left_cube_img = append_images([padding_img, face_images[get_left_face(front_face)]], direction='vertical')
    right_cube_img = append_images([padding_img, face_images[get_right_face(front_face)]], direction='vertical')
    # add_left_cube_img = append_images([padding_img, face_images[get_left_face(front_face)], vertical_cube_img], aligment='top')
    # add_right_cube_img = append_images([padding_img, add_left_cube_img,face_images[get_right_face(front_face)]], aligment='top')
    cube_img = append_images([left_cube_img, vertical_cube_img, right_cube_img], bg_color=(0, 0, 0))
    cube_img.show('foo')


def create_daisy():
    """
    A daisy is a yellow center with white edges
    :return:
    """


def print_cube():
    for face in cube:
        print(face)
    draw_cube(4)
    print('\n\n')

# initial_cube = copy.deepcopy(cube)
#
# for face in initial_cube:
#     print(face)
# move_right(4)
# print(initial_cube==cube)
# for face in cube:
#     print(face)
# move_right(4)
# print(initial_cube==cube)
# for face in cube:
#     print(face)
# move_right(4)
# print(initial_cube==cube)
# for face in cube:
#     print(face)
# move_right(4)
# print(initial_cube==cube)
# print('-----------------')
# for face in cube:
#     print(face)

