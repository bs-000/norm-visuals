import itertools
import operator
import os
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from plotly.graph_objs.mesh3d import Contour
from plotly.subplots import make_subplots
import unicodedata as ud

k_nearest_norms = 3
greek_letters = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
                 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
                 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
display_interactive = 'interactive'
display_classic = 'classic'
display_classic_detailed = 'classic_detailed'
display_types = [display_classic, display_classic_detailed, display_interactive]


def get_generic_h_scale(num_levels):
    """
    Creates a scale for the hierarchy with given levels. Levels are simply numbered starting with 1.
    Users can also create custom scales, as long as the scale is a list.

    :param num_levels: Number of levels for the hierarchy
    :return: a list of levels, starting with the lowest hierarchy level
    """
    return ['$H_{' + str(i + 1) + '}$' for i in range(num_levels)]


def get_generic_dim_scale(dim, num_entries):
    """
    Creates a scale for the given dimension with given levels. Levels are simply the dimension identifier
    numbered starting with 1.
    Users can also create custom scales, as long as the scale is a list.

    :param dim: The dimension to create the list for, intended for the dimensions defined in NormSystem,
                but works with any string.
    :param num_entries: The number of entries for the dimension
    :return: : a list of entries, starting with the lowest number
    """
    return ['$' + dim.lower() + '_{' + str(i + 1) + '}$' for i in range(num_entries)]


def get_contrary_arrow_display_name_interactive(norm, dim_one=False):
    """
    Generates the text to be displayed when hovering over the contrary arrows in interactive mode.

    :param norm: The correspnding norm
    :param dim_one: indicator, whether this is na onedimensional figure
    :return: The generated display name.
    """
    display_name = 'Appeal to contrary: ' + plotly_latex_fix(norm.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm.starttime + '}')
    return display_name


def add_contrary_arrow_interactive(fig, row, col, x, y, showlegend, texts, display_name, color):
    """
    Draws one contrary arrow in one- or two-dimensional interactive figures.

    :param fig: The figure for plotting
    :param row: The row of the subplot
    :param col: The column of the subplot
    :param x: List of x coordinates with start and endpoint
    :param y: List of y coordinates with start and endpoint
    :param showlegend: Whether to show legend
    :param texts: The texts to be displayed on hover on start and end
    :param display_name: The name of the arrow
    :param color: The color of the arrow
    """
    fig.add_trace(go.Scatter(x=x, y=y,
                             marker=dict(size=10, symbol="arrow-up", angleref="previous"),
                             showlegend=showlegend,
                             hovertemplate='%{text}',
                             text=texts,
                             name=display_name, legendgroup=display_name,
                             line={'color': color, 'dash': 'dash'},
                             ), row=row, col=col)


def plot_contrary_arrows_interactive_1d(fig, row, col, outside, x_vs, norm, showlegend_outside,
                                        norm_index, y_pos_with_offset):
    """
    Plots all contrary arrows in one dimensional interactive figures.

    :param fig: The figure for plotting
    :param row: The row of the subplot
    :param col: The column of the subplot
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param x_vs: x_min, x_max, x_start_line, x_end_line, scale
    :param norm: The corresponding norm
    :param showlegend_outside: Whether the legend should be plotted
    :param norm_index: The index of the norm in outside
    :param y_pos_with_offset: The y_position of the arrows
    """
    x_min, x_max, x_start_line, x_end_line, scale = x_vs
    display_name = get_contrary_arrow_display_name_interactive(norm=norm, dim_one=True)
    legend_shown = False
    if x_max != x_end_line:
        add_contrary_arrow_interactive(fig=fig, row=row, col=col, showlegend=showlegend_outside,
                                       display_name=display_name,
                                       color=NormSystem.contrary_map[norm.norm_type],
                                       texts=[plotly_latex_fix(
                                  scale[x_end_line]) + '<extra>' + display_name + '</extra>',
                                     plotly_latex_fix(
                                         scale[x_max - 1]) + '<extra>' + display_name + '</extra>', ],
                                       x=[x_end_line + 0.5, x_max + 0.5],
                                       y=[y_pos_with_offset, y_pos_with_offset])
        legend_shown = True
        outside[norm_index] = True

    if x_min != x_start_line:
        add_contrary_arrow_interactive(fig=fig, row=row, col=col, showlegend=(showlegend_outside and not legend_shown),
                                       display_name=display_name,
                                       color=NormSystem.contrary_map[norm.norm_type],
                                       texts=[plotly_latex_fix(scale[x_start_line - 1]) + '<extra>' + display_name + '</extra>',
                                     plotly_latex_fix(scale[x_min]) + '<extra>' + display_name + '</extra>'],
                                       x=[x_start_line + 0.5, x_min + 0.5],
                                       y=[y_pos_with_offset, y_pos_with_offset])

        outside[norm_index] = True


def plot_contrary_arrows_interactive_2d(fig, col, row, norm, outside, x_vs, y_vs, norm_index, showlegend_outside):
    """
    Plots all contrary arrows in two-dimensional interactive figures.

    :param fig: The figure for plotting
    :param col: The column of the subplot
    :param row: The row of the subplot
    :param norm: The corresponding norm
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param x_vs: x_min, x_max, x_start_rect, x_end_rect
    :param y_vs: y_min, y_max, y_start_rect, y_end_rect
    :param norm_index: The index of the norm in outside
    :param showlegend_outside: Whether to show the legend
    """
    x_min, x_max, x_start_rect, x_end_rect = x_vs
    y_min, y_max, y_start_rect, y_end_rect = y_vs
    legend_shown = False
    # top
    display_name = get_contrary_arrow_display_name_interactive(norm=norm)
    texts = [display_name + '<extra></extra>'] * 2
    if y_max > y_end_rect:
        for x in range(x_min, x_max):
            # offsets are for the corners
            y_offset = 0
            if x > x_end_rect - 1:  # on the right
                y_offset = -0.5 + min(y_max - y_end_rect, x - (x_end_rect - 1))
            if x < x_start_rect:  # on the left
                y_offset = -0.5 + min(y_max - y_end_rect, x_start_rect - x)
            add_contrary_arrow_interactive(fig=fig, row=row, col=col, showlegend=showlegend_outside and not legend_shown,
                                           display_name=display_name,
                                           color=NormSystem.contrary_map[norm.norm_type],
                                           texts=texts,
                                           x=[x + 1, x + 1],
                                           y=[y_end_rect + y_offset + 0.5, y_max + 0.5])
            legend_shown = True
            outside[norm_index] = True
    # bottom
    if y_min < y_start_rect:
        for x in range(x_min, x_max):
            y_offset = 0
            if x > x_end_rect - 1:  # on the right
                y_offset = -0.5 + min(y_start_rect - y_min, x - (x_end_rect - 1))
            if x < x_start_rect:  # on the left
                y_offset = -0.5 + min(y_start_rect - y_min, x_start_rect - x)
            add_contrary_arrow_interactive(fig=fig, row=row, col=col,
                                           showlegend=showlegend_outside and not legend_shown,
                                           display_name=display_name,
                                           color=NormSystem.contrary_map[norm.norm_type],
                                           texts=texts,
                                           x=[x + 1, x + 1],
                                           y=[y_start_rect - y_offset + 0.5, y_min + 0.5])
            legend_shown = True
            outside[norm_index] = True
    # right
    if x_max > x_end_rect:
        for y in range(y_min, y_max):
            x_offset = 0
            if y > y_end_rect - 1:  # at the top
                x_offset = -0.5 + min(x_max - x_end_rect, y - (y_end_rect - 1))
            if y < y_start_rect:  # at the bottom
                x_offset = - 0.5 + min(x_max - x_end_rect, y_start_rect - y)
            add_contrary_arrow_interactive(fig=fig, row=row, col=col,
                                           showlegend=showlegend_outside and not legend_shown,
                                           display_name=display_name,
                                           color=NormSystem.contrary_map[norm.norm_type],
                                           texts=texts,
                                           x=[x_end_rect + x_offset + 0.5, x_max + 0.5],
                                           y=[y + 1, y + 1])
            legend_shown = True
            outside[norm_index] = True

    # left
    if x_min < x_start_rect:
        for y in range(y_min, y_max):
            x_offset = 0
            if y > y_end_rect - 1:  # at the top
                x_offset = -0.5 + min(x_start_rect - x_min, y - (y_end_rect - 1))
            if y < y_start_rect:  # at the bottom
                x_offset = - 0.5 + min(x_start_rect - x_min, y_start_rect - y)
            add_contrary_arrow_interactive(fig=fig, row=row, col=col,
                                           showlegend=showlegend_outside and not legend_shown,
                                           display_name=display_name,
                                           color=NormSystem.contrary_map[norm.norm_type],
                                           texts=texts,
                                           x=[x_start_rect - x_offset + 0.5, x_min + 0.5],
                                           y=[y + 1, y + 1])
            legend_shown = True
            outside[norm_index] = True


def plot_contrary_arrows_interactive_3d(figure, norm, dims):
    """
    Plots the contrary arrows for three-dimensional interactive figures. Draws lines with cones at the end.

    :param figure: Figure for plotting
    :param norm: The corresponding norm
    :param dims: The dimensions for plotting
    """

    # get directions from outside cases
    norm_borders = [(norm.start_values[dims[i]], norm.end_values[dims[i]]) for i in range(len(dims))]

    cases_x = set()
    cases_y = set()
    cases_z = set()
    for case in norm.outside_cases:
        cases_x.add(case.coordinates[dims[0]])
        cases_y.add(case.coordinates[dims[1]])
        cases_z.add(case.coordinates[dims[2]])

    case_dims = [sorted(cases_x, key=lambda x_v: x_v[1]), sorted(cases_y, key=lambda x_v: x_v[1]),
                 sorted(cases_z, key=lambda x_v: x_v[1])]

    x,y,z = [],[],[]
    # draw one arrow per direction (two per dimension at max)
    for dim in range(len(case_dims)):
        for pos in [0, -1]:
            if pos ==0 and norm_borders[dim][pos][1]>case_dims[dim][pos][1]:
                # arrow before start
                if dim == 0 : # x
                    x.append(norm_borders[0][0][1]-0.5)
                    y.append(((norm_borders[1][1][1] - norm_borders[1][0][1]) / 2) + norm_borders[1][0][1])
                    z.append(((norm_borders[2][1][1] - norm_borders[2][0][1]) / 2) + norm_borders[2][0][1])

                    x.append(norm_borders[0][0][1]-1)
                    y.append(((norm_borders[1][1][1]-norm_borders[1][0][1])/2)+ norm_borders[1][0][1])
                    z.append(((norm_borders[2][1][1]-norm_borders[2][0][1])/2)+ norm_borders[2][0][1])
                elif dim == 1: # y
                    x.append(((norm_borders[0][1][1] - norm_borders[0][0][1]) / 2) + norm_borders[0][0][1])
                    y.append(norm_borders[1][0][1]-0.5)
                    z.append(((norm_borders[2][1][1] - norm_borders[2][0][1]) / 2) + norm_borders[2][0][1])

                    x.append(((norm_borders[0][1][1]-norm_borders[0][0][1])/2)+ norm_borders[0][0][1])
                    y.append(norm_borders[1][0][1]-1)
                    z.append(((norm_borders[2][1][1]-norm_borders[2][0][1])/2)+ norm_borders[2][0][1])
                else : # z
                    x.append(((norm_borders[0][1][1] - norm_borders[0][0][1]) / 2) + norm_borders[0][0][1])
                    y.append(((norm_borders[1][1][1] - norm_borders[1][0][1]) / 2) + norm_borders[1][0][1])
                    z.append(norm_borders[2][0][1]-0.5)

                    x.append(((norm_borders[0][1][1]-norm_borders[0][0][1])/2)+ norm_borders[0][0][1])
                    y.append(((norm_borders[1][1][1]-norm_borders[1][0][1])/2)+ norm_borders[1][0][1])
                    z.append(norm_borders[2][0][1]-1)

                # disconnect arrows
                x.append(None)
                y.append(None)
                z.append(None)

            if pos == -1 and norm_borders[dim][pos][1]<case_dims[dim][pos][1]:
                # arrow after end
                if dim == 0 : # x
                    x.append(norm_borders[0][1][1]+0.5)
                    y.append(((norm_borders[1][1][1] - norm_borders[1][0][1]) / 2) + norm_borders[1][0][1])
                    z.append(((norm_borders[2][1][1] - norm_borders[2][0][1]) / 2) + norm_borders[2][0][1])

                    x.append(norm_borders[0][1][1]+1)
                    y.append(((norm_borders[1][1][1]-norm_borders[1][0][1])/2)+ norm_borders[1][0][1])
                    z.append(((norm_borders[2][1][1]-norm_borders[2][0][1])/2)+ norm_borders[2][0][1])
                elif dim == 1: # y
                    x.append(((norm_borders[0][1][1] - norm_borders[0][0][1]) / 2) + norm_borders[0][0][1])
                    y.append(norm_borders[1][1][1]+0.5)
                    z.append(((norm_borders[2][1][1] - norm_borders[2][0][1]) / 2) + norm_borders[2][0][1])

                    x.append(((norm_borders[0][1][1]-norm_borders[0][0][1])/2)+ norm_borders[0][0][1])
                    y.append(norm_borders[1][1][1]+1)
                    z.append(((norm_borders[2][1][1]-norm_borders[2][0][1])/2)+ norm_borders[2][0][1])
                else : # z
                    x.append(((norm_borders[0][1][1] - norm_borders[0][0][1]) / 2) + norm_borders[0][0][1])
                    y.append(((norm_borders[1][1][1] - norm_borders[1][0][1]) / 2) + norm_borders[1][0][1])
                    z.append(norm_borders[2][1][1]+0.5)

                    x.append(((norm_borders[0][1][1]-norm_borders[0][0][1])/2)+ norm_borders[0][0][1])
                    y.append(((norm_borders[1][1][1]-norm_borders[1][0][1])/2)+ norm_borders[1][0][1])
                    z.append(norm_borders[2][1][1]+1)

                # disconnect arrows
                x.append(None)
                y.append(None)
                z.append(None)
    display_name = get_contrary_arrow_display_name_interactive(norm=norm)
    if len(x)>0:
        # plot lines
        color = NormSystem.contrary_map[norm.norm_type]
        figure.add_scatter3d(x=x, y=y, z=z,
                     marker=dict(size=1, color=color),
                     line=dict(color=color,
                               width=20),
                     showlegend=False,
                     opacity=0.5,
                     hovertemplate='%{text}<extra></extra>',
                     text=[display_name for _ in range(len(x))],
                     name=display_name,
                     legendgroup=display_name
                     )
        # plot cones
        u, v, w = [],[],[]
        xc, yc, zc = [], [], []
        for i in range(0, len(x), 3):
            u.append(x[i+1]-x[i])
            v.append(y[i+1]-y[i])
            w.append(z[i+1]-z[i])
            xc.append(x[i+1])
            yc.append(y[i+1])
            zc.append(z[i+1])
        figure.add_cone(x=xc, y=yc, z=zc,
                        u=u, v=v, w=w,
                        showlegend=True,
                        showscale=False,
                        anchor='tail',
                        opacity=0.5,
                        name=display_name,
                        hovertemplate='%{text}',
                        text=[display_name for _ in range(len(xc))],
                        colorscale=[[0, color], [1, color]],
                        hoverlabel=dict(bgcolor=color),
                        legendgroup=display_name
                        )


def add_contrary_arrow(ax, start, length, color):
    """
    Draws an appeal to contrary arrow.

    :param ax: axes of the plot
    :param start: (x_start, y_start) as the start coordinates of the arrow
    :param length: (x_length, y_length) the lengt in each dimension of the arrow
    :param color: the color for the arrow
    """
    x_start, y_start = start
    x_length, y_length = length
    ax.annotate('', xytext=(x_start, y_start), xy=(x_start + x_length, y_start + y_length),
                arrowprops=dict(arrowstyle="-|>", color=color, linestyle=NormSystem.dashed_linestyle,
                                linewidth=NormSystem.linewidth),
                size=18, zorder=10)


def annotate_norm(ax, norm, x, y, dim_one=False):
    """
    Adds the name of the norm to the plot.

    :param ax: Axes of the plot.
    :param norm: The norm to name.
    :param x: x-coordinate for the name.
    :param y: y-coordinate for the name.
    :param dim_one: Whether it's for a one dimensional plot.
    """
    text = '$' + norm.identifier + '$'
    if dim_one:  # add starttime for one dimensional plots
        text = r'$' + norm.identifier + r'_{\mathit{' + norm.starttime + '}}$'
    ax.annotate(text=text,
                xy=(x, y), xytext=(x + NormSystem.annotation_offset, y),
                verticalalignment='center', fontsize=NormSystem.fontsize)


def get_norm_display_name_interactive(norm, dim_one=False):
    """
    Gets the name to be displayed for norms in interactive figures

    :param norm: The correspnding norm
    :param dim_one: Whether this is a one dimensional figure
    :return: The resulting text
    """
    display_name = plotly_latex_fix(norm.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm.starttime + '}')
    return  display_name


def plot_norm_interactive_1d(fig, norm,  y_position, dim, showlegend, row, col):
    """
    Plots a norm in interactive one dimensional figures.

    :param fig: Figure for plotting
    :param norm: The norm
    :param y_position: y_position for the norm
    :param dim: the dimension to be displayed
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    norm_color = NormSystem.colors[norm.norm_type]

    display_name = get_norm_display_name_interactive(norm, True)
    fig.add_trace(go.Scatter(
        x=[norm.start_values[dim][1] + 0.5, norm.end_values[dim][1] + 1.5],
        y=[y_position, y_position],
        hovertemplate='%{text}',
        text=[plotly_latex_fix(norm.start_values[dim][0]) + '<extra>' + display_name + '</extra>',
              plotly_latex_fix(norm.end_values[dim][0]) + '<extra>' + display_name + '</extra>'],
        showlegend=showlegend,
        line={'color': norm_color},
        legendgroup=norm.identifier,
        mode='lines',
        name=display_name,
    ), row=row, col=col)


def plot_norm_interactive_2d(fig, norm, dim_x, dim_y, showlegend, row, col):
    """
    Plots a norm in interactive two-dimensional figures.

    :param fig: Figure for plotting
    :param norm: the norm
    :param dim_x: The x dimension
    :param dim_y: The y dimension
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    norm_color = NormSystem.colors[norm.norm_type]

    display_name = get_norm_display_name_interactive(norm, False)
    x = [norm.start_values[dim_x][1] + 0.5,
         norm.end_values[dim_x][1] + 1.5,
         norm.end_values[dim_x][1] + 1.5,
         norm.start_values[dim_x][1] + 0.5,
         norm.start_values[dim_x][1] + 0.5]
    y = [norm.start_values[dim_y][1] + 0.5,
         norm.start_values[dim_y][1] + 0.5,
         norm.end_values[dim_y][1] + 1.5,
         norm.end_values[dim_y][1] + 1.5,
         norm.start_values[dim_y][1] + 0.5]
    texts = ['<i>' + dim_x + ': ' + plotly_latex_fix(xx) + '<br>'
             + dim_y + ': ' + plotly_latex_fix(yx) + '</i>'
             for (xx, yx) in zip([norm.start_values[dim_x][0],
                                  norm.end_values[dim_x][0],
                                  norm.end_values[dim_x][0],
                                  norm.start_values[dim_x][0],
                                  norm.start_values[dim_x][0]],
                                 [norm.start_values[dim_y][0],
                                  norm.start_values[dim_y][0],
                                  norm.end_values[dim_y][0],
                                  norm.end_values[dim_y][0],
                                  norm.start_values[dim_y][0]]
                                 )]
    fig.add_trace(go.Scatter(x=x, y=y,
                             marker=dict(size=1, color=norm_color),
                             line=dict(color=norm_color,
                                       width=NormSystem.linewidth),
                             showlegend=showlegend,
                             fill="toself",
                            # opacity=0.5,
                             hovertemplate='%{text}<extra></extra>',
                             text=texts,
                             name=display_name,
                             legendgroup=display_name,
                             ),
                  row=row, col=col)


def plot_norm_interactive_3d(figure, norm, dims):
    """
    Plots a norm in three-dimensional interactive figures.

    :param figure: Figure for plotting
    :param norm: The norm
    :param dims: the dimensions
    """
    dim_x, dim_y, dim_z = dims
    norm_color = NormSystem.colors[norm.norm_type]
    x, y, z = [], [], []
    verts_reduced = [reduce_4d_point(vert, (dim_x, dim_y, dim_z)) for vert in norm.vertices]
    verts_reduced = list(set(verts_reduced))

    vert_map = {dim_x:{},dim_y:{},dim_z:{}}
    for i in range(len(verts_reduced)):
        vert = verts_reduced[i]
        x.append(vert[0][1])
        y.append(vert[1][1])
        z.append(vert[2][1])
        vert_map[dim_x][vert[0][1]] =plotly_latex_fix(vert[0][0])
        vert_map[dim_y][vert[1][1]] =plotly_latex_fix(vert[1][0])
        vert_map[dim_z][vert[2][1]] =plotly_latex_fix(vert[2][0])

    x_adjusted = [val-0.5  if val==sorted(set(x))[0]  else (val+0.5) for val in x]
    y_adjusted = [val-0.5  if val==sorted(set(y))[0]  else (val+0.5) for val in y]
    z_adjusted = [val-0.5  if val==sorted(set(z))[0]  else (val+0.5) for val in z]

    # x, y, z now hold start and end positions on the three axis
    # extend them for overlapped plotting
    faces_reduced = [reduce_4d_point(face, (dim_x, dim_y, dim_z)) for face in norm.faces]
    # remove duplicates: two dimensions must change
    faces_reduced = [(x, y, z) for (x, y, z) in faces_reduced if len(x) + len(y) + len(z) == 5]
    # remove duplicates: complete duplicates
    final_faces = []
    [final_faces.append(val) for val in faces_reduced if val not in final_faces]


    # check verts with no change
    # adjust end values
    if len(verts_reduced) == 1:  # no change in any dimension
        # eins anpassen, dann geht es theoretisch ins nächste if ...
        x = x + x
        y = y + y
        z = z + z

        # adjust nodes
        x_adjusted = x_adjusted + [xx + 1 for xx in x_adjusted]
        y_adjusted = y_adjusted + y_adjusted
        z_adjusted = z_adjusted + z_adjusted
        verts_reduced += [((name, val + 1), y, z) for (name, val), y, z in verts_reduced]
    if len(verts_reduced) == 2: # only one value in two dimensions
        x = x + x
        y = y + y
        z = z + z
        if len(set(x_adjusted)) ==1: # adjust x
            # adjust nodes
            x_adjusted = x_adjusted + [xx + 1 for xx in x_adjusted]
            y_adjusted = y_adjusted + y_adjusted
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [((name, val + 1), y, z) for (name, val), y, z in verts_reduced]
        else: # adjust y
            # adjust nodes
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + [yx + 1 for yx in y_adjusted]
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [(x, (name, val + 1), z) for x, (name, val), z in verts_reduced]

        # new face
        new_face = [set(a) for a in zip(*verts_reduced)]

        final_faces = [tuple(new_face)]
    if len(verts_reduced) == 4:  # only one value in one dimension
        x_v, y_v, z_v = final_faces[0]
        x_vs, y_vs, z_vs = x_v.copy(), y_v.copy(), z_v.copy()

        new_final_faces = final_faces.copy()
        x= x+x
        y= y+y
        z= z+z
        # adjust nodes
        if len(x_vs) == 1:
            x_adjusted = x_adjusted + [xx + 1 for xx in x_adjusted]
            y_adjusted = y_adjusted + y_adjusted
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [((name,val+1),y,z) for (name, val),y,z in verts_reduced]
            # opposite face
            new_final_faces += [({(list(x_set)[0][0], list(x_set)[0][1] + 1)}, y, z)
                                for x_set, y, z in final_faces]
            # side faces
            x_vs.add((list(x_vs)[0][0], list(x_vs)[0][1] + 1))
            for y_v in y_vs:
                new_final_faces += [(x_vs, {y_v}, z_vs) for _, y, z in final_faces]
            for z_v in z_vs:
                new_final_faces += [(x_vs, y_vs, {z_v}) for _, y, z in final_faces]
        if len(y_vs) == 1:
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + [yx + 1 for yx in y_adjusted]
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [(x,(name,val+1),z) for x,(name,val), z in verts_reduced]
            # opposite face
            new_final_faces += [(x, {(list(y_set)[0][0], list(y_set)[0][1] + 1)}, z)
                                for x, y_set, z in final_faces]
            # side faces
            y_vs.add((list(y_vs)[0][0], list(y_vs)[0][1] + 1))
            for x_v in x_vs:
                new_final_faces += [({x_v}, y_vs, z_vs) for x, _, z in final_faces]
            for z_v in z_vs:
                new_final_faces += [(x_vs, y_vs, {z_v}) for x, _, z in final_faces]
        if len(z_vs) == 1:
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + y_adjusted
            z_adjusted = z_adjusted + [zx + 1 for zx in z_adjusted]
            verts_reduced += [(x,y,(z_name,z_val+1)) for x,y,(z_name, z_val) in verts_reduced]
            # opposite face
            new_final_faces += [(x, y, {(list(z_set)[0][0], list(z_set)[0][1] + 1)}) for x,y,z_set in final_faces]
            # side faces
            z_vs.add((list(z_vs)[0][0], list(z_vs)[0][1]+1))
            for x_v in x_vs:
                new_final_faces += [({x_v}, y_vs, z_vs) for x, y, _ in final_faces]
            for y_v in y_vs:
                new_final_faces += [(x_vs, {y_v}, z_vs) for x, y, _ in final_faces]

        final_faces = new_final_faces

    i, j, k = [], [], []
    for face in final_faces:
        face_verts = list(itertools.product(face[0], face[1], face[2], ))
        if len(face_verts) == 4:
            # there are still too many triangles rendered... all 4 instead of just 2
            face_verts = [verts_reduced.index(vert) for vert in face_verts]
            face_triangles = list(itertools.combinations(face_verts, 3))
            for tri_i, tri_j, tri_k in face_triangles:
                i.append(tri_i)
                j.append(tri_j)
                k.append(tri_k)
    display_name = get_norm_display_name_interactive(norm)
    figure.add_mesh3d(
        # 8 vertices of a cube
        x=x_adjusted, y=y_adjusted, z=z_adjusted,
        color=norm_color,
        # i, j and k give the vertices of triangles, cube is drawn from triangles
        i=i, j=j, k=k,
        name=display_name,
        opacity=0.2,
        hovertemplate='%{text}',
        text=['<i>'+dim_x+': '+vert_map[dim_x][xx]+'<br>'
              +dim_y+': '+vert_map[dim_y][yx]
              +'<br>'+dim_z+': '+vert_map[dim_z][zx]+'</i>'
              for (xx,yx,zx) in zip(x,y,z)],
        contour=Contour(width=10, color='#ff0000'),
        showlegend=True,
        flatshading=True,
    )


def plot_case_interactive(fig, y, x, identifier, row, col, showlegend):
    """
    Plots a case in interactive one and two-dimensional figure

    :param fig: Figure of the plot
    :param y: y-coordinate
    :param x: x-coordinate
    :param identifier: name / identifier of the case
    :param col: column of the subplot
    :param row: row of the subplot
    :param showlegend: Whether to show the legend
    """
    fig.add_trace(go.Scatter(x=[x[1]+1], y=[y[1]+1], name=plotly_latex_fix(identifier), showlegend=showlegend,
                    hovertemplate='%{text}', legendgroup='case_'+identifier,
                    text=[plotly_latex_fix(identifier)],
                      mode='markers', marker=dict(color='#000000', size=6)), row=row, col=col)
    return


def plot_case_interactive_3d(fig, case, dims):
    """
    Plots a case in a thre dimensional interactive figure.

    :param fig: Figure for plotting
    :param case: The case
    :param dims: The plotted dimensions
    """
    dim_x, dim_y, dim_z = dims
    fig.add_scatter3d(x=[case.coordinates[dim_x][1]], y=[case.coordinates[dim_y][1]],
                      z=[case.coordinates[dim_z][1]], name=plotly_latex_fix(case.identifier),
                      mode='markers', marker=dict(color='#000000', size=6))


def plot_case(ax, y, x, annotations, identifier):
    """
    Plots a case in the figure and adds the intended annotation for the name to the list of existing planned annotations

    :param ax: Axes of the plot
    :param y: y-coordinate
    :param x: x-coordinate
    :param annotations: dict of existing points to annotate
    :param identifier: name / identifier of the case
    :return: the updated annotations
    """
    x_coord = x + 0.5
    y_coord = y + 0.5
    ax.plot(x_coord, y_coord, 'ko', zorder=100)
    if (x_coord, y_coord) not in annotations:
        annotations[(x_coord, y_coord)] = [identifier]
    else:
        annotations[(x_coord, y_coord)].append(identifier)
    return annotations


def annotate_cases(ax, annotations):
    """
    Annotates all cases in the figure. Cases / Annotations at the same coordinates are displayed with a '=' between them

    :param ax: Axes of the plot.
    :param annotations: {(x-coordinate,y-coordiante):[name1, ...nameN]} of annotations for coordiantes.
    """
    for coords in annotations.keys():
        x_coo, y_coo = coords
        text = ' = '.join(annotations[coords])
        ax.annotate(text=text, xytext=(x_coo + 0.1, y_coo + 0.1),
                    xy=(x_coo, y_coo), fontsize=NormSystem.fontsize)


def prepare_hierarchy_axis(ax, scale):
    """
    Sets up hierarchy axis.

    :param ax: axes of the plot
    :param scale: scale with names
    """
    # create scale with ticks in middle
    ax.set_yticks([i - 0.5 for i in range(len(scale) + 1)])
    ax.yaxis.set_tick_params(length=0)  # length 0 to hide ticks
    ax.set_yticklabels([''] + scale, fontsize=NormSystem.fontsize)

    # draw deviders
    for h in range(len(scale) + 1):
        ax.axhline(y=h, color='grey', linestyle='dotted')


def prepare_hierarchy_axis_interactive(fig, scale, row, col):
    """
    Sets up hierarchy axis for interactive one dimensional figures.

    :param fig: figure of the plot
    :param scale: scale with names
    :param row: row of the plot
    :param col: column of the plot
    """
    prepare_one_axis_interactive(fig=fig, scale=scale, x_axis=False, axis_label='', row=row, col=col)

    # draw deviders
    for h in range(len(scale) + 1):
        fig.add_hline(y=h+0.5, line_color='lightgrey', line_dash='dot', row=row, col=col)


def prepare_3d_axes(fig, x_vs, y_vs, z_vs):
    """
    Sets up thre three axis for plotting thre dimensional interactive figures.

    :param fig: The figure for plotting
    :param x_vs: x_label, x_values
    :param y_vs: y_label, y_values
    :param z_vs: z_label, z_values
    """
    x_label, x_values = x_vs
    y_label, y_values = y_vs
    z_label, z_values = z_vs

    fig.update_layout(
        scene=dict(
            xaxis=get_3d_axis_dict(values=x_values, label=x_label),
            yaxis=get_3d_axis_dict(values=y_values, label=y_label),
            zaxis=get_3d_axis_dict(values=z_values, label=z_label),
        ),
    )


def get_3d_axis_dict(values, label):
    """
    Generates the dict for one axis for a three-dimensional interactive figure.

    :param values: The readable vakes for the scales
    :param label: The label of the axis
    :return: The created dict
    """
    ticktexts = [plotly_latex_fix(val) for val in values]
    return dict(
        ticktext=ticktexts,
        title=plotly_latex_fix(label),
        nticks=len(values),
        tickmode='array',
        showline=False,
        range=[-0.5, len(values)+0.5],
        autorange=False,
        zeroline=False,
        tickvals=[i for i in range(len(values))],
    )


def prepare_one_axis_interactive(fig, scale, axis_label, x_axis, row, col):
    """
    Sets up an axis for interactive one- or two-dimensional plotting.

    :param fig: figure of the plot
    :param scale: scale with the names of the axis ticks.
    :param axis_label: Name / label of the axis
    :param x_axis: True if this is the x-axis, False for y-axis
    :param row: row of the subplot
    :param col: column of the subplot
    """
    if x_axis:
        fig.update_xaxes(
            title_text=axis_label,
            ticktext=scale,
            autorange=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            nticks=len(scale),
            tickmode='array',
            range=[0, len(scale)+1],
            tickvals=[i+1 for i in range(len(scale))],
            row=row,
            col=col
        )
    else:
        fig.update_yaxes(
            title_text=axis_label,
            ticktext=scale,
            autorange=False,
            showgrid=False,
            showline=True,
            nticks=len(scale),
            tickmode='array',
            range=[0, len(scale) + 1],
            tickvals=[i+1 for i in range(len(scale))],
            row=row, col=col
        )


def prepare_one_axis(ax, scale, axis_label, x_axis):
    """
    Sets up an axis for plotting.

    :param ax: exes of the plot
    :param scale: scale with the names of the axis ticks.
    :param axis_label: Name / label of the axis
    :param x_axis: Trueif this is the x-axis, False for y-axis
    """
    if x_axis:
        ax.set_xlim((-0.5, len(scale)))
    else:
        ax.set_ylim((-0.5, len(scale)))
    label = '$' + axis_label + '$'
    # create scale with ticks in middle
    if x_axis:
        ax.set_xticks([i + 0.5 for i in range(len(scale))])
        ax.set_xticklabels(scale, fontsize=NormSystem.fontsize)
        ax.set_xlabel(label, loc="right", labelpad=20, fontsize=NormSystem.fontsize)
    else:
        ax.set_yticks([i + 0.5 for i in range(len(scale))])
        ax.set_yticklabels(scale, fontsize=NormSystem.fontsize)
        ax.set_ylabel(label, loc='top', labelpad=20, rotation=0,
                      fontsize=NormSystem.fontsize)

    # adjust scale so last tick isn't on arrow
    if x_axis:
        xlim_left, xlim_right = ax.get_xlim()
        if (xlim_right - xlim_left) <= len(scale) + 0.5:
            ax.set_xlim(left=xlim_left, right=xlim_right + 0.5)
    else:
        ylim_bottom, ylim_top = ax.get_ylim()
        if (ylim_top - ylim_bottom) <= len(scale) + 0.5:
            ax.set_ylim(bottom=ylim_bottom, top=ylim_top + 0.5)


def add_dashed_v_line(ax, x, y_end):
    """
    Adds a vertical line (dashed) to the plot. Always starts on axis.

    :param ax: Axes of the plot
    :param x: x coordinate for the line
    :param y_end: end y coordinate of the line
    """
    ax.vlines(x=x, ymin=-0.5, ymax=y_end, color='k', linestyle=NormSystem.dashed_linestyle,
              zorder=0, linewidth=NormSystem.linewidth)


def add_dashed_h_line(ax, x_end, y):
    """
    Adds a horizontal line (dashed) to the plot. Always starts on axis.

    :param ax: axes of the plot
    :param x_end: x coordinate on where to ende the line
    :param y: y coordinate for the line
    """
    ax.hlines(y=y, xmax=x_end, xmin=-0.5, color='k', linestyle=NormSystem.dashed_linestyle,
              zorder=0, linewidth=NormSystem.linewidth)


def add_analogy_h_line(ax, x_start, x_end, y, color):
    """
    Adds a horizontal line for one dimensional plots for analogies.

    :param ax: Axes of the plot.
    :param x_start: x coordinate to start.
    :param x_end: x coordinate to end.
    :param y: y coordinate of the line
    :param color: color of the line.
    """
    ax.hlines(y=y, xmax=x_end, xmin=x_start, color=color, linestyle=NormSystem.dashed_linestyle,
              zorder=0, linewidth=NormSystem.linewidth)


def add_analogy_rectangle(ax, color, xy, width, height):
    """
    Adds an analogy rectangle for two-dimensional plots.

    :param ax: axes of the plot.
    :param color: color for the rectangle
    :param xy: x and y coordinate to start point
    :param width: width of the rectangle
    :param height: height of the rectangle
    """
    ax.add_patch(
        Rectangle(xy, width=width, height=height,
                  facecolor='none',
                  linestyle=NormSystem.dashed_linestyle,
                  linewidth=NormSystem.linewidth,
                  edgecolor=color))


def get_bounding_box_range(dim, norm, case):
    """
    Calculates start and end value for analogy box in one dimension for three-dimensional plotting.

    :param dim: The dimension to calculate bounding values for
    :param norm: The norm
    :param case: The case
    :return: start, end
    """
    values = sorted({case.coordinates[dim], norm.start_values[dim], norm.end_values[dim]}, key=lambda x: x[1])
    return values[0],values[-1]


def swap_coordinate(to_swap, swap_data, swap_index):
    """
    For the triangles in three-dimensional plotting: swaps one point of a coordinate point.

    :param to_swap: The data to swap, dictionary
    :param swap_data: The data to swap from
    :param swap_index: The new data index to insert from swap_data
    :return: the swapped data
    """
    swap_copy = to_swap.copy()
    if len(set(swap_data[swap_index])) == 1: # only one possible value
        swap_value = swap_data[swap_index][0]
    else: # now we only have two possible values
        swap_value = [x for x in swap_data[swap_index] if x != to_swap[swap_index]][0]
    swap_copy[swap_index] = swap_value
    return swap_copy


def get_analogy_display_name_interactive(norm, dim_one=False, case=None):
    """
    Generates the text to be displayed for an analogy in interactive figures.

    :param norm: The corresponding norm
    :param dim_one: Whether this is for a one dimensional figure
    :param case: The case this analogy belongs to
    :return: The generated display name
    """
    display_name = 'Analogy: ' + plotly_latex_fix(norm.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm.starttime + '}')
    if case is not None:
        display_name += ' to ' + plotly_latex_fix(case.identifier)
    return display_name


def plot_analogy_interactive_1d(fig, norm, row, col, outside, norm_index, x_vs, y_position, showlegend_outside):
    """
    Plots an analogy in a one dimensional interactive figure.

    :param fig: The figure for plotting
    :param norm: The corresponding norm
    :param row: row of the subplot
    :param col: column of the subplot
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param norm_index: Index of the norm in outside
    :param x_vs: x_min, x_start_line, x_max, x_end_line, scale
    :param y_position: the y position for plotting
    :param showlegend_outside: Whether to show the legend
    """
    x_min, x_start_line, x_max, x_end_line, scale = x_vs
    display_name = get_analogy_display_name_interactive(norm=norm, dim_one=True)
    x_vs, y_vs, texts = [], [], []
    if x_min != x_start_line:
        x_vs += [x_min + 0.5, x_start_line + 0.5, None, ]
        y_vs += [y_position, y_position, None, ]
        texts += [plotly_latex_fix(scale[x_min]) + '<extra>' + display_name + '</extra>',
                  plotly_latex_fix(scale[x_start_line - 1]) + '<extra>' + display_name + '</extra>',
                  None]
        outside[norm_index] = True
    if x_max != x_end_line:
        x_vs += [x_end_line + 0.5, x_max + 0.5]
        y_vs += [y_position, y_position]
        texts += [plotly_latex_fix(scale[x_end_line]) + '<extra>' + display_name + '</extra>',
                  plotly_latex_fix(scale[x_max - 1]) + '<extra>' + display_name + '</extra>', ]
        outside[norm_index] = True

    fig.add_trace(go.Scatter(
        x=x_vs,
        y=y_vs,
        hovertemplate='%{text}',
        text=texts,
        showlegend=showlegend_outside,
        legendgroup=display_name,
        line={'color': NormSystem.colors[norm.norm_type], 'dash': 'dash'},
        mode='lines',
        name=display_name,
    ), row=row, col=col)


def plot_analogy_interactive_2d(fig, norm, x_vs, y_vs, showlegend, row, col):
    """
    Plots an analogy in two-dimensional interactive figures.

    :param fig: Figure for plotting
    :param norm: The corresponding norm
    :param x_vs: dim_x, x_min, x_max, x_scale
    :param y_vs: dim_y, y_min, y_max, y_scale
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    dim_x, x_min, x_max, x_scale = x_vs
    dim_y, y_min, y_max, y_scale = y_vs
    norm_color = NormSystem.colors[norm.norm_type]
    display_name = get_analogy_display_name_interactive(norm)
    x = [x_min, x_max, x_max, x_min, x_min]
    y = [y_min, y_min, y_max, y_max, y_min]
    texts = ['<i>' + dim_x + ': ' + plotly_latex_fix(x_scale[int(xx)]) + '<br>'
             + dim_y + ': ' + plotly_latex_fix(y_scale[int(yx)]) + '</i>'
             for (xx, yx) in zip([x_min -0.5, x_max -1.5, x_max-1.5,
                                  x_min -0.5, x_min -0.5],
                                 [y_min -0.5, y_min -0.5, y_max -1.5,
                                  y_max -1.5, y_min -0.5]
                                 )]
    fig.add_trace(go.Scatter(x=x, y=y,
                             marker=dict(size=1, color=norm_color),
                             line=dict(color=norm_color, dash='dash',
                                       width=NormSystem.linewidth),
                             showlegend=showlegend,
                             opacity=0.5,
                             hovertemplate='%{text}<extra></extra>',
                             text=texts,
                             name=display_name,
                             legendgroup=display_name,
                             ),
                  row=row, col=col)


def plot_analogy_interactive_3d(figure, norm, dims, case):
    """
    Plots an analogy in three-dimensional interactive figures.

    :param figure: The figure for plotting
    :param norm: The corresponding norm
    :param dims: The dimensions to plot
    :param case: The case this analogy belongs to
    """
    dim_x, dim_y, dim_z = dims
    norm_color = NormSystem.colors[norm.norm_type]
    # get start and end coordinates for the bounding box
    bounding_values = get_bounding_box_range(dim=dim_x, norm=norm, case=case), \
                      get_bounding_box_range(dim=dim_y, norm=norm, case=case),\
                      get_bounding_box_range(dim=dim_z, norm=norm, case=case)
    vertices = [x for x in itertools.product(*bounding_values)]
    vert_map = {dim_x: {v[0][1]:plotly_latex_fix(v[0][0]) for v in vertices},
                dim_y: {v[1][1]:plotly_latex_fix(v[1][0]) for v in vertices},
                dim_z: {v[2][1]:plotly_latex_fix(v[2][0]) for v in vertices}}

    x_lines = list()
    y_lines = list()
    z_lines = list()

    x_vs, y_vs, z_vs = bounding_values
    for startpoint in [[x_vs[0], y_vs[0], z_vs[0]],[x_vs[1], y_vs[1], z_vs[1]]]:
        for change_id_one, change_id_two in [x for x in itertools.combinations([0,1,2], 2)]:
            # first triangle
            point_one = startpoint
            #change one
            point_two = swap_coordinate(to_swap=startpoint, swap_data=bounding_values, swap_index = change_id_one)
            #change the other
            point_three = swap_coordinate(to_swap=startpoint, swap_data=bounding_values, swap_index = change_id_two)
            #change both
            point_four = swap_coordinate(to_swap=point_three, swap_data=bounding_values, swap_index = change_id_one)

            for point in [point_one, point_two, point_four, point_three, point_one]:
                x_v, y_v, z_v = point
                x_lines.append(x_v[1])
                y_lines.append(y_v[1])
                z_lines.append(z_v[1])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)


    x = [v[0][1] for v in vertices]
    y = [v[1][1] for v in vertices]
    z = [v[2][1] for v in vertices]

    texts = [None if xx is None else
             '<i>' + dim_x + ': ' + vert_map[dim_x][xx] + '<br>'
             + dim_y + ': ' + vert_map[dim_y][yx]
             + '<br>' + dim_z + ': ' + vert_map[dim_z][zx] + '</i>'
             for (xx, yx, zx) in zip(x_lines, y_lines, z_lines)]

    x_lines = [None if val is None else val-0.5  if val==sorted(set(x))[0]  else (val+0.5) for val in x_lines]
    y_lines = [None if val is None else val-0.5  if val==sorted(set(y))[0]  else (val+0.5) for val in y_lines]
    z_lines = [None if val is None else val-0.5  if val==sorted(set(z))[0]  else (val+0.5) for val in z_lines]

    figure.add_scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        hovertemplate='%{text}',
        text=texts,
        line={'dash': 'dash', 'color': norm_color},
        mode='lines',
        name=get_analogy_display_name_interactive(norm=norm, case=case),
        opacity=0.5,
    )

def change_color(norm_type, color_hex):
    """
    Changes the color for the given normative type to the given hex code

    :param norm_type: One of the three norm types
    :param color_hex: A hexadecimal code for a color
    """
    if norm_type in [NormSystem.type_obl, NormSystem.type_prohib, NormSystem.type_perm]:
        NormSystem.colors[norm_type] = color_hex
        NormSystem.contrary_map = {NormSystem.type_prohib: NormSystem.colors[NormSystem.type_perm],
                                   NormSystem.type_perm: NormSystem.colors[NormSystem.type_prohib]}


def reduce_4d_point(point, dims):
    """
    Removes unneccesary dimension from a four dimensional point for thre dimensional plotting.

    :param point: The point
    :param dims: The dimensions to plot
    :return: The reduced point
    """
    return (point[NormSystem.dimensions.index(dims[0])], point[NormSystem.dimensions.index(dims[1])],
           point[NormSystem.dimensions.index(dims[2])])


def plotly_latex_fix(latex_raw_text):
    """
    Plotly can't write latex in many places. This method converts the latex to html style for printing.
    https://plotly.com/chart-studio-help/adding-HTML-and-links-to-charts/

    :param latex_raw_text: Raw latex string
    :return: text converted to use html tags
    """
    new_text = latex_raw_text.replace('$', '')
    while '\\mathfrak' in new_text:
        split_text_start, split_text_end = new_text.split('\\mathfrak')
        if split_text_end[0]=='{':
            sub_split = split_text_end[1:].split('}')
            split_text_end = sub_split[0] + '}'.join(sub_split[1:])
        new_text = split_text_start+split_text_end
    for sign, start, end in [('_', '<sub>', '</sub>'), ('^', '<sup>', '</sup>'),
                             ('\\', '&', ';')]:
        split_text = new_text.split(sign)
        new_text = split_text[0]
        for i in range(1,len(split_text)):
            curr_text = split_text[i]
            if curr_text.startswith('{'):
                subscript_split = curr_text[1:].split('}')
                subscript = subscript_split[0]
                curr_text = '}'.join(subscript_split[1:])
            else:
                subscript_split = curr_text.split(' ')
                subscript = subscript_split[0]
                curr_text = ' '.join(subscript_split[1:])

            if subscript.lower() in greek_letters and sign == '\\':
                capital = 'CAPITAL' if subscript[0].isupper() else 'SMALL'
                greek_letter_code = 'MATHEMATICAL ITALIC '+capital +' ' + subscript.upper()
                new_text = new_text + ud.lookup(greek_letter_code) + ' ' + curr_text
            else:
                new_text = new_text + start+ subscript + end +' '+ curr_text
    new_text = '<i>'+new_text+'</i>'
    return new_text


def setup_interactive_design(fig):
    """
    Sets up general layout for interactive figures.

    :param fig: The figure
    """
    fig.update_layout(
        font_family="Times New Roman",
        font_size=14,
        title_font_family="Times New Roman",
        hoverlabel=dict(font=dict(family='Times New Roman')),
        template="plotly_white",
    )
    fig.update_layout(scene_aspectmode='data')


def reset_colors():
    """
    Resets the colors of the norms to the original color setting.
    """
    NormSystem.colors = {NormSystem.type_obl: '#9cbad8', NormSystem.type_perm: '#fcde78',
                         NormSystem.type_prohib: '#e66560'}
    NormSystem.contrary_map = {NormSystem.type_prohib: NormSystem.colors[NormSystem.type_perm],
                               NormSystem.type_perm: NormSystem.colors[NormSystem.type_prohib]}


def create_dir(path):
    """
    Creates a directory if it doesn't exist yet.

    :param path: The path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


class NormSystem:
    dim_o = 'O'
    dim_r = 'R'
    dim_s = 'S'
    dim_t = 'T'
    hierarchy = 'H'
    type_obl = 'obligation'
    type_perm = 'permission'
    type_prohib = 'prohibition'
    fontsize = 18
    dashed_linestyle = (0, (7, 5))
    linewidth = 2
    annotation_offset = 0.25
    colors = {type_obl: '#9cbad8', type_perm: '#fcde78', type_prohib: '#e66560'}
    # possible hatches: https://stackoverflow.com/questions/14279344/how-can-i-add-textures-to-my-bars-and-wedges
    hatches = {type_obl: '*', type_perm: '//', type_prohib: '\\\\'}
    contrary_map = {type_prohib: colors[type_perm], type_perm: colors[type_prohib]}
    dimensions = [dim_o, dim_r, dim_s, dim_t]

    def __init__(self, object_scale, r_scale, subject_scale, time_scale, hierarchy_scale):
        """
        Initializes the attributes.

        :param object_scale: scale values for dim o
        :param r_scale: scale values for dim r
        :param subject_scale: scale values for dim s
        :param time_scale: scale values for dim t
        :param hierarchy_scale: scale values for the hierarchy scale
        """
        self.scales = {NormSystem.dim_o: object_scale, NormSystem.dim_r: r_scale,
                       NormSystem.dim_s: subject_scale, NormSystem.dim_t: time_scale}
        self.h_scale = hierarchy_scale
        self.norm_list = []
        self.case_list = []

    def get_dim_vals(self, raw_vals, dim_id):
        """
        From the given textual values the mathematical values are derived, the values ordered and returned.

        :param raw_vals: Textual values for a dimension (two)
        :param dim_id: The dimension.
        :return: A list containing a tuple of mathematical and textual start and end points.
        """
        start, end = raw_vals
        start_pos, end_pos = self.get_position(start, dim_id), self.get_position(end, dim_id)
        if start_pos > end_pos:
            start, end = end, start
            start_pos, end_pos = end_pos, start_pos
        return [(start, start_pos), (end, end_pos)]

    def add_norm(self, o_vals, r_vals, s_vals, t_vals, hierarchy, starttime, norm_type, identifier):
        """
        Method for adding a norm to the system. Inclusive for the end values.

        :param o_vals: (start, end) values for dimension o from scale
        :param r_vals: (start, end) values for dimension r from scale
        :param s_vals: (start, end) values for dimension s from scale
        :param t_vals: (start, end) values for dimension t from scale
        :param hierarchy: Hierarchy of the norm
        :param starttime: indicator for when norm was introduced
        :param norm_type: One of the Norm types of NormSystem
        :param identifier: Name of the norm
        """
        self.norm_list.append(Norm(o_vals=self.get_dim_vals(raw_vals=o_vals, dim_id=NormSystem.dim_o),
                                   r_vals=self.get_dim_vals(raw_vals=r_vals, dim_id=NormSystem.dim_r),
                                   s_vals=self.get_dim_vals(raw_vals=s_vals, dim_id=NormSystem.dim_s),
                                   t_vals=self.get_dim_vals(raw_vals=t_vals, dim_id=NormSystem.dim_t),
                                   hierarchy=(hierarchy, self.h_scale.index(hierarchy)), starttime=starttime,
                                   norm_type=norm_type,
                                   identifier=identifier))

    def add_case(self, o_val, r_val, s_val, t_val, identifier=None):
        """
        Adds a case to the norm system.

        :param o_val: o coordinate
        :param r_val: r coordinate
        :param s_val: s coordinate
        :param t_val: t coordinate
        :param identifier: Name of the norm. If non is given, it will be C_x with x the current count of cases
                            in the NormSystem
        """
        if identifier is None:
            case_id = 1
            identifier = r'$\mathfrak{C}_{' + str(case_id) + '}$'
            case_names = [case.identifier for case in self.case_list]
            while identifier in case_names:
                case_id += 1
                identifier = r'$\mathfrak{C}_{' + str(case_id) + '}$'

        self.case_list.append(Case(o_val=(o_val, self.get_position(o_val, NormSystem.dim_o)),
                                   r_val=(r_val, self.get_position(r_val, NormSystem.dim_r)),
                                   s_val=(s_val, self.get_position(s_val, NormSystem.dim_s)),
                                   t_val=(t_val, self.get_position(t_val, NormSystem.dim_t)),
                                   identifier=identifier
                                   ))

    def delete_norm(self, norm):
        """
        Deletes a norm from the list of norms.

        :param norm: The norm to delete.
        """
        self.norm_list.remove(norm)

    def delete_case(self, case):
        """
        Deletes a case from the list of cases.

        :param case: The case to delete.
        """
        self.case_list.remove(case)

    def get_relevant_norms(self):
        """
        Is called when a figure in a NormSystem should be plotted. Selects all normas relevant to the cases.
        These are all norms which subsume a case and in case that a case is not subsumed by any norm, the k closest
        norms are returned for that norm.

        :return: a set of all relevant norms.
        """
        for norm in self.norm_list:
            norm.outside_cases = []
        result_list = set()
        for case in self.case_list:
            fitting_norms = []
            for norm in self.norm_list:
                if norm.subsumes(case):
                    fitting_norms.append(norm)
            if len(fitting_norms) == 0 and len(self.norm_list) > 0:  # no subsumption exists
                norm_distances = [(norm.calculate_min_distance(case), norm) for norm in self.norm_list]
                norm_distances = sorted(norm_distances, reverse=True, key=operator.itemgetter(0))
                fitting_norms = norm_distances[:k_nearest_norms]
                fitting_norms = [norm for _, norm in fitting_norms]
            for norm in fitting_norms:
                norm.add_outside_case(case)
            result_list = result_list.union(fitting_norms)

        return result_list

    def get_relevant_norms_sorted_by_hierarchy(self):
        """
        Returns the relevant norm and sorts them by hierarchy (for one dimensional plots)

        :return: A list of len(hierarchy_scale) with each position containing a list of norms in that hierarchy level
        """
        norms = self.get_relevant_norms()

        sorted_norms = []
        for h_name in self.h_scale:
            h_norms = [norm for norm in norms if norm.hierarchy[0] == h_name]
            sorted_norms.append(h_norms)

        return sorted_norms

    def get_position(self, value, dimension):
        """
        Gets the mathematical value for the given literal value on a scale.

        :param value: Value to get mathematical value for.
        :param dimension: dimension from NormSystem.dimensions
        :return:
        """
        scale = self.scales[dimension]
        return scale.index(value)

    def draw_one_dim_subplot(self, ax, dim, detailed):
        """
        Draws one subplot for a one dimensional plot.

        :param ax: axes of the plot
        :param dim: x-axis from NormSystem.dimensions
        :param detailed: whether to draw the detailed version
        """
        scale = self.scales[dim]

        # cases
        annotations = {}
        for case in self.case_list:
            annotations = plot_case(ax, x=case.coordinates[dim][1], y=-1,
                                    annotations=annotations, identifier=case.identifier)
        annotate_cases(ax, annotations)

        # norms
        relevant_norms = self.get_relevant_norms_sorted_by_hierarchy()
        for norms in relevant_norms:
            num_norms = len(norms)
            offset = 1 / (num_norms + 1)
            for i in range(len(norms)):
                current_offset = offset * (i + 1)
                norm = norms[i]
                norm_color = NormSystem.colors[norm.norm_type]
                y_position = norm.hierarchy[1] + current_offset
                ax.hlines(y=y_position, xmax=norm.end_values[dim][1] + 1, xmin=norm.start_values[dim][1],
                          color=norm_color,
                          linewidth=NormSystem.linewidth)

                # dashed lines from axes
                dashes_lines_x_points = [norm.start_values[dim][1], norm.end_values[dim][1] + 1]

                x_start_line = x_min = norm.start_values[dim][1]
                x_end_line = x_max = norm.end_values[dim][1] + 1
                # outside
                if len(norm.outside_cases) > 0:
                    for case in norm.outside_cases:
                        x_case = case.coordinates[dim][1]
                        x_max = max(x_case + 1, x_max)
                        x_min = min(x_case, x_min)
                    if x_min < x_start_line:
                        dashes_lines_x_points.append(x_min)
                    if x_max > x_end_line:
                        dashes_lines_x_points.append(x_max)

                    if norm.norm_type in NormSystem.contrary_map:
                        add_contrary_arrow(ax, start=(x_start_line, y_position - (0.3 * offset)),
                                           length=(x_min - x_start_line, 0),
                                           color=NormSystem.contrary_map[norm.norm_type])
                        add_contrary_arrow(ax, start=(x_end_line, y_position - (0.3 * offset)),
                                           length=(x_max - x_end_line, 0),
                                           color=NormSystem.contrary_map[norm.norm_type])

                    add_analogy_h_line(ax=ax, x_start=x_min, x_end=x_start_line, color=norm_color, y=y_position)
                    add_analogy_h_line(ax=ax, x_start=x_end_line, x_end=x_max, color=norm_color, y=y_position)

                annotate_norm(ax=ax, norm=norm, x=x_max, y=y_position, dim_one=True)
                # draw dashed lines from axes
                if detailed:
                    for x_point in dashes_lines_x_points:
                        add_dashed_v_line(ax=ax, x=x_point, y_end=y_position)

        # axis
        ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines[['bottom']].set_linewidth(NormSystem.linewidth)
        ax.set_xlim((-0.5, len(scale)))
        prepare_one_axis(scale=scale, axis_label=dim, ax=ax, x_axis=True)
        prepare_hierarchy_axis(scale=self.h_scale, ax=ax)

    def draw_one_dim_subplot_interactive(self, fig, dim, row, col, outside):
        """
        Draws one subplot for a one dimensional plot (interactive).

        :param fig: figure of the plot
        :param dim: x-axis from NormSystem.dimensions
        :param row: row of the subplot
        :param col: column of the subplot
        :param outside: List with truth values indicating whether the legend should be plotted (for updating)
        :return: The updated outside
        """
        scale = self.scales[dim]
        showlegend = (col+row) == 2 # only show legend for first plot
        setup_interactive_design(fig)

        # cases
        for case in self.case_list:
            plot_case_interactive(fig, x=case.coordinates[dim], y=['',-0.65], identifier=case.identifier, row=row, col=col, showlegend=showlegend)

        # norms
        relevant_norms = self.get_relevant_norms_sorted_by_hierarchy()
        for norms in relevant_norms:
            num_norms = len(norms)
            offset = 1 / (num_norms + 1)

            for i in range(len(norms)):
                current_offset = offset * (i + 1)
                norm = norms[i]
                y_position = norm.hierarchy[1] + current_offset + 0.5

                plot_norm_interactive_1d(fig=fig, norm=norm, y_position=y_position, dim=dim, showlegend=showlegend, row=row, col=col)

                x_start_line = x_min = norm.start_values[dim][1]
                x_end_line = x_max = norm.end_values[dim][1] + 1
                # outside
                if len(norm.outside_cases) > 0:
                    showlegend_outside = showlegend | (not outside[self.norm_list.index(norm)])  # show analogy etc. legend only if it wasn't plotted before
                    for case in norm.outside_cases:
                        x_case = case.coordinates[dim][1]
                        x_max = max(x_case + 1, x_max)
                        x_min = min(x_case, x_min)

                    # contrary arrows
                    if norm.norm_type in NormSystem.contrary_map:
                        plot_contrary_arrows_interactive_1d(fig=fig, row=row, col=col, outside=outside,
                                                            x_vs=(x_min, x_max, x_start_line, x_end_line, scale),
                                                            norm=norm, showlegend_outside=showlegend_outside,
                                                            norm_index=self.norm_list.index(norm),
                                                            y_pos_with_offset=y_position - (0.3 * offset))

                    # analogy
                    plot_analogy_interactive_1d(fig=fig, norm=norm, row=row, col=col, outside=outside,
                                                norm_index= self.norm_list.index(norm),
                                                x_vs=(x_min, x_start_line, x_max, x_end_line, scale),
                                                y_position=y_position, showlegend_outside=showlegend_outside)

        # axis
        prepare_one_axis_interactive(scale=scale, axis_label=dim, fig=fig, x_axis=True, row=row, col=col)
        prepare_hierarchy_axis_interactive(scale=self.h_scale, fig=fig, row=row, col=col)
        return outside

    def draw_two_dim_subplot_interactive(self, fig, dim_x, dim_y, row, col, outside):
        """
        Draws one subplot for a two-dimensional plot.

        :param fig: figure of the plot
        :param dim_x: x-axis from NormSystem.dimensions
        :param dim_y: y-axis from NormSystem.dimensions
        :param row: row of the subplot
        :param col: column of the subplot
        :param outside: List with truth values indicating whether the legend should be plotted (for updating)
        """
        showlegend = (col + row) == 2  # only show legend for first plot
        setup_interactive_design(fig)

        # norms
        for norm in self.get_relevant_norms():
            plot_norm_interactive_2d(fig=fig, norm=norm, dim_x=dim_x, dim_y=dim_y, showlegend=showlegend, col=col, row=row)

            if len(norm.outside_cases) > 0:
                showlegend_outside = showlegend | (not outside[self.norm_list.index(norm)])  # show analogy etc. legend only if it wasn't plotted before

                x_start_rect = x_min = norm.start_values[dim_x][1]
                y_start_rect = y_min = norm.start_values[dim_y][1]
                y_end_rect = y_max = norm.end_values[dim_y][1] + 1
                x_end_rect = x_max = norm.end_values[dim_x][1] + 1
                for case in norm.outside_cases:
                    y_case = case.coordinates[dim_y][1]
                    x_case = case.coordinates[dim_x][1]
                    y_max = max(y_case + 1, y_max)
                    y_min = min(y_case, y_min)
                    x_max = max(x_case + 1, x_max)
                    x_min = min(x_case, x_min)

                # contrary arrows
                if norm.norm_type in NormSystem.contrary_map:
                    plot_contrary_arrows_interactive_2d(fig=fig, col=col, row=row, norm=norm, outside=outside,
                                                        x_vs = (x_min, x_max, x_start_rect, x_end_rect),
                                                        y_vs = (y_min, y_max, y_start_rect, y_end_rect),
                                                        norm_index=self.norm_list.index(norm), showlegend_outside=showlegend_outside)

                # analogy
                plot_analogy_interactive_2d(fig=fig, row=row, col=col, showlegend=showlegend,
                                            x_vs=(dim_x, x_min+0.5,x_max+0.5, self.scales[dim_x]),
                                            y_vs=(dim_y, y_min+0.5,y_max+0.5, self.scales[dim_y]),
                                            norm=norm)

        # cases, down here to plot above the rest
        for case in self.case_list:
            plot_case_interactive(fig, x=case.coordinates[dim_x], y=case.coordinates[dim_y], identifier=case.identifier, row=row, col=col,
                                  showlegend=showlegend)

        prepare_one_axis_interactive(scale=self.scales[dim_x], axis_label=dim_x, fig=fig, x_axis=True, row=row, col=col)
        prepare_one_axis_interactive(scale=self.scales[dim_y], axis_label=dim_y, fig=fig, x_axis=False, row=row, col=col)
        return outside

    def draw_two_dim_subplot(self, ax, dim_x, dim_y, detailed):
        """
        Draws one subplot for a two-dimensional plot.

        :param ax: axes of the plot
        :param dim_x: x-axis from NormSystem.dimensions
        :param dim_y: y-axis from NormSystem.dimensions
        :param detailed: whether to draw the detailed version
        """
        # cases
        annotations = {}
        for case in self.case_list:
            annotations = plot_case(ax, x=case.coordinates[dim_x][1], y=case.coordinates[dim_y][1],
                                    annotations=annotations, identifier=case.identifier)
        annotate_cases(ax, annotations)

        # norms
        norm_annotations = {}
        for norm in self.get_relevant_norms():
            norm_color = NormSystem.colors[norm.norm_type]

            ax.add_patch(Rectangle((norm.start_values[dim_x][1], norm.start_values[dim_y][1]),
                                   norm.end_values[dim_x][1] - norm.start_values[dim_x][1] + 1,
                                   norm.end_values[dim_y][1] - norm.start_values[dim_y][1] + 1,
                                   facecolor='none',
                                   hatch=NormSystem.hatches[norm.norm_type], linewidth=NormSystem.linewidth,
                                   edgecolor=norm_color))
            x, y = norm.end_values[dim_x][1] + 1, norm.start_values[dim_y][1] + (norm.end_values[dim_y][1] + 1 -
                                                                                 norm.start_values[dim_y][1]) / 2
            if (x, y) not in norm_annotations:
                norm_annotations[(x, y)] = [norm]
            else:
                norm_annotations[(x, y)].append(norm)

            # dashed lines from axes
            dashed_y_start = norm.start_values[dim_y][1]
            dashed_x_start = norm.start_values[dim_x][1]
            dashes_lines_x_points = [norm.start_values[dim_x][1], norm.end_values[dim_x][1] + 1]
            dashes_lines_y_points = [norm.start_values[dim_y][1], norm.end_values[dim_y][1] + 1]

            if len(norm.outside_cases) > 0:
                x_start_rect = x_min = norm.start_values[dim_x][1]
                y_start_rect = y_min = norm.start_values[dim_y][1]
                y_end_rect = y_max = norm.end_values[dim_y][1] + 1
                x_end_rect = x_max = norm.end_values[dim_x][1] + 1
                for case in norm.outside_cases:
                    y_case = case.coordinates[dim_y][1]
                    x_case = case.coordinates[dim_x][1]
                    y_max = max(y_case + 1, y_max)
                    y_min = min(y_case, y_min)
                    x_max = max(x_case + 1, x_max)
                    x_min = min(x_case, x_min)
                # contrary arrows
                if norm.norm_type in NormSystem.contrary_map:
                    # top
                    if y_max > y_end_rect:
                        for x in range(x_min, x_max):
                            # offsets are for the corners
                            y_offset = 0
                            if x > x_end_rect - 1:  # on the right
                                y_offset = -0.5 + min(y_max - y_end_rect, x - (x_end_rect - 1))
                            if x < x_start_rect:  # on the left
                                y_offset = -0.5 + min(y_max - y_end_rect, x_start_rect - x)
                            add_contrary_arrow(ax, start=(x + 0.5, y_end_rect + y_offset),
                                               length=(0, y_max - y_end_rect - y_offset),
                                               color=NormSystem.contrary_map[norm.norm_type])
                    # bottom
                    if y_min < y_start_rect:
                        for x in range(x_min, x_max):
                            y_offset = 0
                            if x > x_end_rect - 1:  # on the right
                                y_offset = -0.5 + min(y_start_rect - y_min, x - (x_end_rect - 1))
                            if x < x_start_rect:  # on the left
                                y_offset = -0.5 + min(y_start_rect - y_min, x_start_rect - x)
                            add_contrary_arrow(ax, start=(x + 0.5, y_start_rect - y_offset),
                                               length=(0, y_min - y_start_rect + y_offset),
                                               color=NormSystem.contrary_map[norm.norm_type])
                    # right
                    if x_max > x_end_rect:
                        for y in range(y_min, y_max):
                            x_offset = 0
                            if y > y_end_rect - 1:  # at the top
                                x_offset = -0.5 + min(x_max - x_end_rect, y - (y_end_rect - 1))
                            if y < y_start_rect:  # at the bottom
                                x_offset = - 0.5 + min(x_max - x_end_rect, y_start_rect - y)
                            add_contrary_arrow(ax, start=(x_end_rect + x_offset, y + 0.5),
                                               length=(x_max - x_end_rect - x_offset, 0),
                                               color=NormSystem.contrary_map[norm.norm_type])
                    # left
                    if x_min < x_start_rect:
                        for y in range(y_min, y_max):
                            x_offset = 0
                            if y > y_end_rect - 1:  # at the top
                                x_offset = -0.5 + min(x_start_rect - x_min, y - (y_end_rect - 1))
                            if y < y_start_rect:  # at the bottom
                                x_offset = - 0.5 + min(x_start_rect - x_min, y_start_rect - y)
                            add_contrary_arrow(ax, start=(x_start_rect - x_offset, y + 0.5),
                                               length=(x_min - x_start_rect + x_offset, 0),
                                               color=NormSystem.contrary_map[norm.norm_type])

                    dashed_y_start = y_min
                    dashed_x_start = x_min

                # analogy
                if x_min < x_start_rect:
                    dashes_lines_x_points.append(x_min)
                if x_max > x_end_rect:
                    dashes_lines_x_points.append(x_max)
                if y_min < y_start_rect:
                    dashes_lines_y_points.append(y_min)
                if y_max > y_end_rect:
                    dashes_lines_y_points.append(y_max)
                height = y_max - y_min
                width = x_max - x_min
                add_analogy_rectangle(ax=ax, width=width, color=norm_color,
                                      height=height,
                                      xy=(x_min, y_min))

            # draw dashed lines
            if detailed:
                for y_point in dashes_lines_y_points:
                    add_dashed_h_line(ax=ax, x_end=dashed_x_start, y=y_point)
                for x_point in dashes_lines_x_points:
                    add_dashed_v_line(ax=ax, x=x_point, y_end=dashed_y_start)

        for key in norm_annotations.keys():
            norms = norm_annotations[key]
            x, y = key
            for i in range(len(norms)):
                annotate_norm(ax=ax, norm=norms[i], x=x, y=y+(i*0.25))

        # arrows on axes
        ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False)
        ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(NormSystem.linewidth)
        prepare_one_axis(scale=self.scales[dim_x], axis_label=dim_x, ax=ax, x_axis=True)
        prepare_one_axis(scale=self.scales[dim_y], axis_label=dim_y, ax=ax, x_axis=False)

    def draw_three_dim_subplot(self, fig, dim_to_exclude):
        """
        Draws a three-dimensional plot.

        :param fig: The figure
        :param dim_to_exclude: The dimension not to display
        """
        dims = NormSystem.dimensions.copy()
        dims.remove(dim_to_exclude)
        dim_x, dim_y, dim_z = dims[0], dims[1], dims[2]
        setup_interactive_design(fig)

        prepare_3d_axes(fig=fig, x_vs=(dim_x, self.scales[dim_x]),
                        y_vs=(dim_y, self.scales[dim_y]), z_vs=(dim_z, self.scales[dim_z]))

        # cases
        for case in self.case_list:
            plot_case_interactive_3d(fig=fig, case=case, dims=(dim_x, dim_y, dim_z))

        # norms
        for norm in self.get_relevant_norms():
            plot_norm_interactive_3d(figure=fig, norm=norm, dims=dims)

            # analogies
            for case in norm.outside_cases:
                plot_analogy_interactive_3d(figure=fig, norm=norm, dims=dims, case=case)

            # contrary arrows
            if norm.norm_type in NormSystem.contrary_map:
                plot_contrary_arrows_interactive_3d(figure=fig, norm=norm, dims=dims)

    def draw_dims_one_all(self, display_type=display_classic_detailed, figure=None, saveidentifier=None):
        """
        Coordinates drawing figures for one dimenions each.

        :param saveidentifier: an identifier to include in the name of the saved file.
        :param figure: figure to use for plotting
        :param display_type:  Indicates which version to draw for the figures.
        """
        if display_type is not display_interactive:
            if saveidentifier is not None:
                fig = plt.figure(figsize=(15, 10))
            else:
                fig = figure

            if fig is not None:
                matplotlib.rcParams['mathtext.fontset'] = 'cm'
                ax1 = fig.add_subplot(221)  # Plot with: 4 rows, 1 column
                self.draw_one_dim_subplot(ax1, NormSystem.dim_o, display_type==display_classic_detailed)
                ax2 = fig.add_subplot(222)
                self.draw_one_dim_subplot(ax2, NormSystem.dim_r, display_type==display_classic_detailed)
                ax3 = fig.add_subplot(223)
                self.draw_one_dim_subplot(ax3, NormSystem.dim_s, display_type==display_classic_detailed)
                ax4 = fig.add_subplot(224)
                self.draw_one_dim_subplot(ax4, NormSystem.dim_t, display_type==display_classic_detailed)

                fig.tight_layout()
            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_dims_one.png'
                plt.savefig(savename)
                plt.close()
        else:
            fig = make_subplots(rows=2, cols=2)
            outside = [False]*len(self.norm_list)
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_o, row=1, col=1, outside=outside)
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_r, row=1, col=2, outside=outside)
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_s, row=2, col=1, outside=outside)
            self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_t, row=2, col=2, outside=outside)

            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_dims_one.html'
                fig.write_html(savename, include_mathjax='cdn')

    def draw_dims_one(self, detailed, x_dim, fig):
        """
        Coordinates drawing figures for one dimenions each.

        :param x_dim: dimension to display on x-axis
        :param fig: figure to use for plotting
        :param detailed:  Whether to draw the detailled version of the figures.
        """
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        ax1 = fig.add_subplot(111)
        self.draw_one_dim_subplot(ax1, x_dim, detailed)
        fig.tight_layout()

    def draw_dims_two(self, dim_x, dim_y, detailed, fig):
        """
        Coordinates drawing figures for two dimenions each.

        :param dim_x: x-axis (from NormSystem.dimensions)
        :param dim_y: y-axis (from NormSystem.dimensions)
        :param fig: figure for plotting
        :param detailed: Whether to draw the detailled version of the figures.
        """
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        ax1 = fig.add_subplot(111)  # Plot with: 1 row, 2 column, first subplot.
        self.draw_two_dim_subplot(ax1, dim_x, dim_y, detailed)
        fig.tight_layout()

    def draw_dims_two_all(self, dims_one, dims_two, display_type=display_classic_detailed, saveidentifier=None, figure=None):
        """
        Coordinates drawing figures for two dimenions each.

        :param dims_one: (dim_x_str, dim_y_str) for first plot (from NormSystem.dimensions)
        :param dims_two: (dim_x_str, dim_y_str) for second plot (from NormSystem.dimensions)
        :param figure: figure for plotting
        :param saveidentifier: an identifier to include in the name of the saved file.
        :param display_type: Indicator for the version to plot
        """
        if display_type != display_interactive:
            if saveidentifier is not None:
                fig = plt.figure(figsize=(20, 10))
            else:
                fig = figure

            if fig is not None:
                matplotlib.rcParams['mathtext.fontset'] = 'cm'
                ax1 = fig.add_subplot(121)  # Plot with: 1 row, 2 column, first subplot.
                self.draw_two_dim_subplot(ax1, dims_one[0], dims_one[1], display_type==display_classic_detailed)
                ax2 = fig.add_subplot(122)  # Plot with: 1 row, 2 column, second subplot.
                self.draw_two_dim_subplot(ax2, dims_two[0], dims_two[1], display_type==display_classic_detailed)
                fig.tight_layout()

            if saveidentifier is not None:
                path = display_type + '/'
                create_dir(path)
                savename = path + saveidentifier + '_dims_two.png'
                plt.savefig(savename)
                plt.close()
        else:
            fig = make_subplots(rows=1, cols=2)
            outside = [False]*len(self.norm_list)
            outside = self.draw_two_dim_subplot_interactive(fig, dims_one[0], dims_one[1], row=1, col=1, outside=outside)
            self.draw_two_dim_subplot_interactive(fig, dims_two[0], dims_two[1], row=1, col=2, outside=outside)

            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_dims_two.html'
                fig.write_html(savename, include_mathjax='cdn')

    def draw_dims_three(self, saveidentifier, dim_to_exclude):
        """
        Wrapper method for plotting a three-dimensional diagram and saving it.

        :param saveidentifier: Identifier for saving
        :param dim_to_exclude: The dimension not to display
        """
        path=display_interactive+'/'
        create_dir(path)
        savename = path + saveidentifier + '_dims_three.html'

        fig = make_subplots(rows=1, cols=1)
        self.draw_three_dim_subplot(fig, dim_to_exclude)
        fig.write_html(savename, include_mathjax='cdn')

    def reset(self):
        """
        Removes all norms and cases from the NormSystem.
        """
        self.norm_list = []
        self.case_list = []


class Case:
    """
    Class for one case.
    """

    def __init__(self, o_val, r_val, s_val, t_val, identifier):
        """
        Initializes the attributes.

        :param o_val: coordinate on dimension o
        :param r_val: coordinate on dimension r
        :param s_val: coordinate on dimension s
        :param t_val: coordinate on dimension t
        :param identifier: Name / Identifier of the case
        """
        self.coordinates = {NormSystem.dim_o: o_val, NormSystem.dim_r: r_val,
                            NormSystem.dim_s: s_val, NormSystem.dim_t: t_val}
        self.identifier = identifier


def gram_schmidt_2(base, new_vector):
    """
    Calculates an ordinal vector for two vectors according to the Gram-Schmidt-method.

    :param base: The base vector.
    :param new_vector: The vector to convert to an ordinal vector.
    :return: The new ordinal vector.
    """
    return new_vector - (base.dot(new_vector) / base.dot(base)) * base


def gram_schmidt_3(base, ordinal_vector, new_vector):
    """
    Calculates an ordinal vector for three vectors according to the Gram-Schmidt-method.

    :param base: The base vector
    :param ordinal_vector: A second vector already ordinal to the base.
    :param new_vector: The vector to convert to an ordinal vector.
    :return: The new ordinal vector.
    """
    return new_vector - ((base.dot(new_vector) / base.dot(base)) * base +
                         (ordinal_vector.dot(new_vector) / ordinal_vector.dot(ordinal_vector)) * ordinal_vector)


def get_mathematical_vertex(vertice_with_names):
    """
    From a vertice with literal coordinates from the scale, creates a mathematical vertex.

    :param vertice_with_names: a 4-tuple of 2-tuples. One tuple for each dimension (4), each with the literal
    coordinate and the mathematical coordinate.
    :return: A mathematical vertex ready for calculations.
    """
    res = np.array([])
    for i in range(len(vertice_with_names)):
        res = np.append(res, [vertice_with_names[i][1]])
    return res


def is_value_in_range(value, val_range):
    """
    Checks whether a value is in a given range.

    :param value: A mathematical value (number)
    :param val_range: A set of tuples (ordered) which define the range. Each tuple includes the literal coordinate
                      name and the mathematical value of the coordinate. Range is inclusive at the end.
    :return: True, if the given value is in the range, False otherwise.
    """
    range_vals = sorted([val for _, val in val_range])
    if len(range_vals) == 1:  # range has only one value
        return value == range_vals[0]
    else:
        return range_vals[0] <= value <= range_vals[1]


def vector_length(vector):
    """
    Calculates the legnth of a vector. (Euclidean Norm)

    :param vector: Vector to calculate length for.
    :return: Calculated length.
    """
    return sum(vector * vector) ** 0.5


class Norm:
    """
    Class for one Norm.
    """

    def __init__(self, o_vals, r_vals, s_vals, t_vals, hierarchy, starttime, norm_type, identifier):
        """
        Norms are hyperrectangles https://de.wikipedia.org/wiki/Hyperrechteck
        (https://de.wikipedia.org/wiki/Hyperw%C3%BCrfel for visual reference on faces and nodes).

        Initializes the attributes, calculates the vertices and faces of the hyperrectangle.

        :param o_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm
                        for dimensiono
        :param r_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm
                        for dimension r
        :param s_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm
                        for dimension s
        :param t_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm
                        for dimension t
        :param hierarchy: hierarchy of the norm
        :param starttime: short indicator on the time when the norm was introduced
        :param norm_type: either NormSystem.type_obl, NormSystem.type_perm or NormSystem.type_prohib
        :param identifier: Name / identifier of the norm
        """
        self.start_values = {NormSystem.dim_o: o_vals[0], NormSystem.dim_r: r_vals[0],
                             NormSystem.dim_s: s_vals[0], NormSystem.dim_t: t_vals[0]}
        self.end_values = {NormSystem.dim_o: o_vals[1], NormSystem.dim_r: r_vals[1],
                           NormSystem.dim_s: s_vals[1], NormSystem.dim_t: t_vals[1]}
        self.hierarchy = hierarchy
        self.starttime = starttime
        self.norm_type = norm_type
        self.identifier = identifier
        self.outside_cases = []

        # calculate vertices
        self.vertices = []
        for o in o_vals:
            for r in r_vals:
                for s in s_vals:
                    for t in t_vals:
                        self.vertices.append((o, r, s, t))

        # calculate faces
        self.faces = []
        for vertice in self.vertices:
            o, r, s, t = vertice
            neighbours = []
            for o_v in o_vals:
                if o != o_v:
                    neighbours.append((o_v, r, s, t))
            for r_v in r_vals:
                if r != r_v:
                    neighbours.append((o, r_v, s, t))
            for s_v in s_vals:
                if s != s_v:
                    neighbours.append((o, r, s_v, t))
            for t_v in t_vals:
                if t != t_v:
                    neighbours.append((o, r, s, t_v))
            for comb in itertools.combinations(neighbours, 2):
                neigh_a, neigh_b = comb
                na_o, na_r, na_s, na_t = neigh_a
                nb_o, nb_r, nb_s, nb_t = neigh_b
                o_range = {o, na_o, nb_o}
                r_range = {r, na_r, nb_r}
                s_range = {s, na_s, nb_s}
                t_range = {t, na_t, nb_t}
                new_face = (o_range, r_range, s_range, t_range)
                found = False
                for face in self.faces:
                    if face == new_face:
                        found = True
                if not found:
                    self.faces.append(new_face)

    def calculate_min_distance(self, case):
        """
        Calculates the minimal distance between a given case and the norm. Uses Euclidean Distance.
        Could maybe be much faster....

        :param case: The case to check the distance for.
        :return: The calculated minimal distance.
        """
        min_dist = None
        case_vector = get_mathematical_vertex((case.coordinates[NormSystem.dim_o],
                                               case.coordinates[NormSystem.dim_r],
                                               case.coordinates[NormSystem.dim_s],
                                               case.coordinates[NormSystem.dim_t]))
        # calculate minimal distance from any face
        if len(self.faces) == 0:  # happens if a norm has equal start and end point in all dimensions
            # distance must be calculated from the point
            norm_vector = get_mathematical_vertex((self.start_values[NormSystem.dim_o],
                                                   self.start_values[NormSystem.dim_r],
                                                   self.start_values[NormSystem.dim_s],
                                                   self.start_values[NormSystem.dim_t]))
            min_dist = vector_length(case_vector - norm_vector)
        for (o_range, r_range, s_range, t_range) in self.faces:
            face_vertices = []
            for o in o_range:
                for r in r_range:
                    for s in s_range:
                        for t in t_range:
                            face_vertices.append((o, r, s, t))

            face_vertices = [get_mathematical_vertex(vert) for vert in face_vertices]
            # vector to transform points to plane system
            stuetz_vector = face_vertices[0]
            # two vectors to define the plane, must be ordinal
            a_vector = face_vertices[1] - stuetz_vector
            b_vector = face_vertices[2] - stuetz_vector
            # vectors should already be ordinal as we have a hyperrectangle
            ordinal_b_vector = gram_schmidt_2(base=a_vector, new_vector=b_vector)

            # ordinal vector from plane to the case
            ordinal_case_vector = gram_schmidt_3(base=a_vector, ordinal_vector=ordinal_b_vector,
                                                 new_vector=case_vector - stuetz_vector)
            # projection of the case on the plane
            projection_case_vector = case_vector - ordinal_case_vector
            distance = np.Inf
            # check if case is in the face on the plane
            if is_value_in_range(value=projection_case_vector[0], val_range=o_range) and \
                    is_value_in_range(value=projection_case_vector[1], val_range=r_range) and \
                    is_value_in_range(value=projection_case_vector[2], val_range=s_range) and \
                    is_value_in_range(value=projection_case_vector[3], val_range=t_range):
                # if inside, then distance is length of ordinal vector
                distance = vector_length(ordinal_case_vector)
            else:
                # otherwise distance must be calculated to borders of the faces (ordinal vector to line)
                for comb in itertools.combinations(face_vertices, 2):
                    vert_a, vert_b = comb
                    # define line vector
                    line_vect = vert_a - vert_b
                    # if neighbours
                    if np.count_nonzero(line_vect != 0) == 1:  # only one change in coordinate, so neighbours
                        # orthogonal projection on a line
                        ordinal_case_vector_to_line = gram_schmidt_2(base=line_vect, new_vector=case_vector - vert_b)
                        projection_case_vector_on_line = case_vector - ordinal_case_vector_to_line
                        # check if projection is petween the vertices
                        o_range_line = sorted([vert_a[0], vert_b[0]])
                        r_range_line = sorted([vert_a[1], vert_b[1]])
                        s_range_line = sorted([vert_a[2], vert_b[2]])
                        t_range_line = sorted([vert_a[3], vert_b[3]])
                        if o_range_line[0] <= projection_case_vector_on_line[0] <= o_range_line[1] and \
                                r_range_line[0] <= projection_case_vector_on_line[1] <= r_range_line[1] and \
                                s_range_line[0] <= projection_case_vector_on_line[2] <= s_range_line[1] and \
                                t_range_line[0] <= projection_case_vector_on_line[3] <= t_range_line[1]:
                            # if yes, then length of the orthogonal vector
                            distance = vector_length(ordinal_case_vector_to_line)
                        else:
                            # else distance to points
                            dist_a = vector_length(case_vector - vert_a)
                            dist_b = vector_length(case_vector - vert_b)
                            if dist_a < dist_b:
                                distance = dist_a
                            else:
                                distance = dist_b
            if min_dist is None or distance < min_dist:
                min_dist = distance
        return min_dist

    def subsumes(self, case):
        """
        Checks if a case is subsumed by the norm. Since it's a hyperrectangle, we can simply check whether the
        coordinates of the case are within start and end value of the norm for every dimension.

        :param case: The case to check.
        :return: True if the norm subsumes the case, False otherwise.
        """
        outside = False
        for dim in NormSystem.dimensions:
            if not self.start_values[dim][1] <= case.coordinates[dim][1] <= self.end_values[dim][1]:
                outside = True
        return not outside

    def add_outside_case(self, case):
        """
        Adds a given case to the list of cases not subsumed by the norm. Used for plotting later on.

        :param case: Case to add.
        """
        self.outside_cases.append(case)
