import numpy as np


def calculate_box_vertices(cx, cy, cz, width, length, height, rot_matrix, apply_rot_matrix=False):
    # calculate half-length, half-width and half-height
    hlength = length / 2
    hwidth = width / 2
    hheight = height / 2

    # calculate vertex coordinates relative to center
    vertices_rel = np.array(
        [
            [hlength, hwidth, hheight],
            [hlength, hwidth, -hheight],
            [hlength, -hwidth, hheight],
            [hlength, -hwidth, -hheight],
            [-hlength, hwidth, hheight],
            [-hlength, hwidth, -hheight],
            [-hlength, -hwidth, hheight],
            [-hlength, -hwidth, -hheight],
        ]
    )

    if apply_rot_matrix:
        # apply rotation matrix to vertices
        vertices_rot = vertices_rel @ rot_matrix.T

        # calculate final vertex coordinates
        vertices = vertices_rot + np.array([cx, cy, cz])

    else:
        vertices = vertices_rel + np.array([cx, cy, cz])

    return vertices


vertexes = calculate_box_vertices(
    cx=19.788434758363664,
    cy=8.789811614313502,
    cz=-0.798320147505247,
    width=1.6056261,
    length=3.8312221,
    height=1.8019634,
    rot_matrix=np.array(
        [[0.02153533, -0.99976809, 0], [0.99976809, 0.02153533, 0], [0.02153533, -0.99976809, 1]]
    ),
)
print(vertexes.T)
