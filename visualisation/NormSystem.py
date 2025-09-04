import itertools
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
interactive_font_size = 16


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


def get_contrary_arrow_display_name_interactive(norm_aspect_value, dim_one=False):
    """
    Generates the text to be displayed when hovering over the contrary arrows in interactive mode.

    :param norm_aspect_value: The correspnding norm_aspect_value
    :param dim_one: indicator, whether this is na onedimensional figure
    :return: The generated display name.
    """
    display_name = 'Appeal to contrary: ' + plotly_latex_fix(norm_aspect_value.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm_aspect_value.starttime + '}')
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
                                        norm_index, y_pos_with_offset, aspect):
    """
    Plots all contrary arrows in one dimensional interactive figures.

    :param fig: The figure for plotting
    :param row: The row of the subplot
    :param col: The column of the subplot
    :param aspect: aspect for plotting
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param x_vs: x_min, x_max, x_start_line, x_end_line, scale
    :param norm: The corresponding norm_aspect_value
    :param showlegend_outside: Whether the legend should be plotted
    :param norm_index: The index of the norm_aspect_value in outside
    :param y_pos_with_offset: The y_position of the arrows
    """
    x_min, x_max, x_start_line, x_end_line, scale = x_vs
    display_name = get_contrary_arrow_display_name_interactive(norm_aspect_value=norm, dim_one=True)
    legend_shown = False
    contrary_color = aspect.get_contrary_norm_color(norm)
    if x_max != x_end_line:
        add_contrary_arrow_interactive(fig=fig, row=row, col=col, showlegend=showlegend_outside,
                                       display_name=display_name,
                                       color=contrary_color,
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
                                       color=contrary_color,
                                       texts=[plotly_latex_fix(scale[x_start_line - 1]) + '<extra>' + display_name + '</extra>',
                                     plotly_latex_fix(scale[x_min]) + '<extra>' + display_name + '</extra>'],
                                       x=[x_start_line + 0.5, x_min + 0.5],
                                       y=[y_pos_with_offset, y_pos_with_offset])

        outside[norm_index] = True


def plot_contrary_arrows_interactive_2d(fig, col, row, norm, outside, x_vs, y_vs, norm_index, showlegend_outside, aspect):
    """
    Plots all contrary arrows in two-dimensional interactive figures.

    :param fig: The figure for plotting
    :param col: The column of the subplot
    :param row: The row of the subplot
    :param aspect: aspect for plotting
    :param norm: The corresponding norm_aspect_value
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param x_vs: x_min, x_max, x_start_rect, x_end_rect
    :param y_vs: y_min, y_max, y_start_rect, y_end_rect
    :param norm_index: The index of the norm_aspect_value in outside
    :param showlegend_outside: Whether to show the legend
    """
    x_min, x_max, x_start_rect, x_end_rect = x_vs
    y_min, y_max, y_start_rect, y_end_rect = y_vs
    legend_shown = False
    contrary_color = aspect.get_contrary_norm_color(norm)
    # top
    display_name = get_contrary_arrow_display_name_interactive(norm_aspect_value=norm)
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
                                           color=contrary_color,
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
                                           color=contrary_color,
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
                                           color=contrary_color,
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
                                           color=contrary_color,
                                           texts=texts,
                                           x=[x_start_rect - x_offset + 0.5, x_min + 0.5],
                                           y=[y + 1, y + 1])
            legend_shown = True
            outside[norm_index] = True


def plot_contrary_arrows_interactive_3d(figure, norm_aspect_value, dims, aspect):
    """
    Plots the contrary arrows for three-dimensional interactive figures. Draws lines with cones at the end.

    :param figure: Figure for plotting
    :param aspect: aspect for plotting
    :param norm_aspect_value: The corresponding norm_aspect_value
    :param dims: The dimensions for plotting
    """
    # get directions from outside cases
    norm_borders = [(norm_aspect_value.start_values[dims[i]], norm_aspect_value.end_values[dims[i]]) for i in range(len(dims))]

    cases_x = set()
    cases_y = set()
    cases_z = set()
    for case in norm_aspect_value.outside_cases:
        cases_x.add(case.coordinates(aspect.aspect_name)[dims[0]])
        cases_y.add(case.coordinates(aspect.aspect_name)[dims[1]])
        cases_z.add(case.coordinates(aspect.aspect_name)[dims[2]])

    case_dims = [sorted(cases_x, key=lambda x_v: x_v.value), sorted(cases_y, key=lambda x_v: x_v.value),
                 sorted(cases_z, key=lambda x_v: x_v.value)]

    x,y,z = [],[],[]
    # draw one arrow per direction (two per dimension at max)
    for dim in range(len(case_dims)):
        for pos in [0, -1]:
            if pos ==0 and norm_borders[dim][pos].value>case_dims[dim][pos].value:
                # arrow before start
                if dim == 0 : # x
                    x.append(norm_borders[0][0].value-0.5)
                    y.append(((norm_borders[1][1].value - norm_borders[1][0].value) / 2) + norm_borders[1][0].value)
                    z.append(((norm_borders[2][1].value - norm_borders[2][0].value) / 2) + norm_borders[2][0].value)

                    x.append(norm_borders[0][0].value-1)
                    y.append(((norm_borders[1][1].value-norm_borders[1][0].value)/2)+ norm_borders[1][0].value)
                    z.append(((norm_borders[2][1].value-norm_borders[2][0].value)/2)+ norm_borders[2][0].value)
                elif dim == 1: # y
                    x.append(((norm_borders[0][1].value - norm_borders[0][0].value) / 2) + norm_borders[0][0].value)
                    y.append(norm_borders[1][0].value-0.5)
                    z.append(((norm_borders[2][1].value - norm_borders[2][0].value) / 2) + norm_borders[2][0].value)

                    x.append(((norm_borders[0][1].value-norm_borders[0][0].value)/2)+ norm_borders[0][0].value)
                    y.append(norm_borders[1][0].value-1)
                    z.append(((norm_borders[2][1].value-norm_borders[2][0].value)/2)+ norm_borders[2][0].value)
                else : # z
                    x.append(((norm_borders[0][1].value - norm_borders[0][0].value) / 2) + norm_borders[0][0].value)
                    y.append(((norm_borders[1][1].value - norm_borders[1][0].value) / 2) + norm_borders[1][0].value)
                    z.append(norm_borders[2][0].value-0.5)

                    x.append(((norm_borders[0][1].value-norm_borders[0][0].value)/2)+ norm_borders[0][0].value)
                    y.append(((norm_borders[1][1].value-norm_borders[1][0].value)/2)+ norm_borders[1][0].value)
                    z.append(norm_borders[2][0].value-1)

                # disconnect arrows
                x.append(None)
                y.append(None)
                z.append(None)

            if pos == -1 and norm_borders[dim][pos].value<case_dims[dim][pos].value:
                # arrow after end
                if dim == 0 : # x
                    x.append(norm_borders[0][1].value+0.5)
                    y.append(((norm_borders[1][1].value - norm_borders[1][0].value) / 2) + norm_borders[1][0].value)
                    z.append(((norm_borders[2][1].value - norm_borders[2][0].value) / 2) + norm_borders[2][0].value)

                    x.append(norm_borders[0][1].value+1)
                    y.append(((norm_borders[1][1].value-norm_borders[1][0].value)/2)+ norm_borders[1][0].value)
                    z.append(((norm_borders[2][1].value-norm_borders[2][0].value)/2)+ norm_borders[2][0].value)
                elif dim == 1: # y
                    x.append(((norm_borders[0][1].value - norm_borders[0][0].value) / 2) + norm_borders[0][0].value)
                    y.append(norm_borders[1][1].value+0.5)
                    z.append(((norm_borders[2][1].value - norm_borders[2][0].value) / 2) + norm_borders[2][0].value)

                    x.append(((norm_borders[0][1].value-norm_borders[0][0].value)/2)+ norm_borders[0][0].value)
                    y.append(norm_borders[1][1].value+1)
                    z.append(((norm_borders[2][1].value-norm_borders[2][0].value)/2)+ norm_borders[2][0].value)
                else : # z
                    x.append(((norm_borders[0][1].value - norm_borders[0][0].value) / 2) + norm_borders[0][0].value)
                    y.append(((norm_borders[1][1].value - norm_borders[1][0].value) / 2) + norm_borders[1][0].value)
                    z.append(norm_borders[2][1].value+0.5)

                    x.append(((norm_borders[0][1].value-norm_borders[0][0].value)/2)+ norm_borders[0][0].value)
                    y.append(((norm_borders[1][1].value-norm_borders[1][0].value)/2)+ norm_borders[1][0].value)
                    z.append(norm_borders[2][1].value+1)

                # disconnect arrows
                x.append(None)
                y.append(None)
                z.append(None)
    display_name = get_contrary_arrow_display_name_interactive(norm_aspect_value=norm_aspect_value)
    if len(x)>0:
        # plot lines
        color = aspect.get_contrary_norm_color(norm_aspect_value)
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


def annotate_norm(ax, norm_aspect_value, x, y,  dim_one=False):
    """
    Adds the name of the norm_aspect_value to the plot.

    :param ax: Axes of the plot.
    :param norm_aspect_value: The norm_aspect_value to name.
    :param x: x-coordinate for the name.
    :param y: y-coordinate for the name.
    :param dim_one: Whether it's for a one dimensional plot.
    """
    identifier_var = norm_aspect_value.identifier.replace('$','')
    if '$' + identifier_var +'$' != norm_aspect_value.identifier:
        identifier_var = norm_aspect_value.identifier
    text = '$' + identifier_var + '$'
    if dim_one:  # add starttime for one dimensional plots
        text = r'$' + identifier_var + r'_{\mathit{' + norm_aspect_value.starttime + '}}$'
    ax.annotate(text=text,
                xy=(x, y), xytext=(x + NormSystem.annotation_offset, y),
                verticalalignment='center', fontsize=NormSystem.fontsize)


def get_norm_display_name_interactive(norm_aspect_value, dim_one=False):
    """
    Gets the name to be displayed for norms in interactive figures

    :param norm_aspect_value: The correspnding norm_aspect_value
    :param dim_one: Whether this is a one dimensional figure
    :return: The resulting text
    """
    display_name = plotly_latex_fix(norm_aspect_value.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm_aspect_value.starttime + '}')
    return  display_name


def plot_norm_interactive_1d(fig, norm_aspect_value, y_position, dim, showlegend, row, col, aspect):
    """
    Plots a norm_aspect_value in interactive one dimensional figures.

    :param fig: Figure for plotting
    :param norm_aspect_value: The norm_aspect_value
    :param aspect: the aspect_name to plot
    :param y_position: y_position for the norm_aspect_value
    :param dim: the dimension to be displayed
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    norm_color = aspect.get_norm_color(norm_aspect_value)
    start = norm_aspect_value.start_values[dim]
    end = norm_aspect_value.end_values[dim]

    display_name = get_norm_display_name_interactive(norm_aspect_value, dim_one=True)
    fig.add_trace(go.Scatter(
        x=[start.value + 0.5, end.value + 1.5],
        y=[y_position, y_position],
        hovertemplate='%{text}',
        text=[plotly_latex_fix(start.name) + '<extra>' + display_name + '</extra>',
              plotly_latex_fix(end.name) + '<extra>' + display_name + '</extra>'],
        showlegend=showlegend,
        line={'color': norm_color},
        legendgroup=norm_aspect_value.identifier,
        mode='lines',
        name=display_name,
    ), row=row, col=col)


def plot_norm_interactive_2d(fig, norm_aspect_value, dim_x, dim_y, showlegend, row, col, aspect):
    """
    Plots a norm_aspect_value in interactive two-dimensional figures.

    :param fig: Figure for plotting
    :param norm_aspect_value: the norm_aspect_value
    :param dim_x: The x dimension
    :param aspect: Aspect to plot
    :param dim_y: The y dimension
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    norm_color = aspect.get_norm_color(norm_aspect_value)
    start = {dim_x: norm_aspect_value.start_values[dim_x],
             dim_y: norm_aspect_value.start_values[dim_y]}
    end = {dim_x: norm_aspect_value.end_values[dim_x],
           dim_y: norm_aspect_value.end_values[dim_y]}

    display_name = get_norm_display_name_interactive(norm_aspect_value, dim_one=False)
    x = [start[dim_x].value + 0.5,
         end[dim_x].value + 1.5,
         end[dim_x].value + 1.5,
         start[dim_x].value + 0.5,
         start[dim_x].value + 0.5]
    y = [start[dim_y].value + 0.5,
         start[dim_y].value + 0.5,
         end[dim_y].value + 1.5,
         end[dim_y].value + 1.5,
         start[dim_y].value + 0.5]
    texts = ['<i>' + dim_x + ': ' + plotly_latex_fix(xx) + '<br>'
             + dim_y + ': ' + plotly_latex_fix(yx) + '</i>'
             for (xx, yx) in zip([start[dim_x].name,
                                  end[dim_x].name,
                                  end[dim_x].name,
                                  start[dim_x].name,
                                  start[dim_x].name],
                                 [start[dim_y].name,
                                  start[dim_y].name,
                                  end[dim_y].name,
                                  end[dim_y].name,
                                  start[dim_y].name]
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


def plot_norm_interactive_3d(figure, norm_aspect_value, dims, aspect):
    """
    Plots a norm_aspect_value in three-dimensional interactive figures.

    :param figure: Figure for plotting
    :param norm_aspect_value: The norm_aspect_value
    :param aspect: The aspect to plot
    :param dims: the dimensions
    """
    dim_x, dim_y, dim_z = dims
    norm_color = aspect.get_norm_color(norm_aspect_value)
    x, y, z = [], [], []
    verts_reduced = [reduce_4d_point(vert, (dim_x, dim_y, dim_z)) for vert in norm_aspect_value.vertices]
    verts_reduced = list(set(verts_reduced))

    vert_map = {dim_x:{},dim_y:{},dim_z:{}}
    for i in range(len(verts_reduced)):
        vert = verts_reduced[i]
        x.append(vert[0].value)
        y.append(vert[1].value)
        z.append(vert[2].value)
        vert_map[dim_x][vert[0].value] =plotly_latex_fix(vert[0].name)
        vert_map[dim_y][vert[1].value] =plotly_latex_fix(vert[1].name)
        vert_map[dim_z][vert[2].value] =plotly_latex_fix(vert[2].name)

    x_adjusted = [val-0.5  if val==sorted(set(x))[0]  else (val+0.5) for val in x]
    y_adjusted = [val-0.5  if val==sorted(set(y))[0]  else (val+0.5) for val in y]
    z_adjusted = [val-0.5  if val==sorted(set(z))[0]  else (val+0.5) for val in z]

    # x, y, z now hold start and end positions on the three axis
    # extend them for overlapped plotting
    faces_reduced = [reduce_4d_point(face, (dim_x, dim_y, dim_z)) for face in norm_aspect_value.faces]
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
        verts_reduced += [(DimValue(name=x.name, value=x.value + 1), y, z) for x, y, z in verts_reduced]
    if len(verts_reduced) == 2: # only one value in two dimensions
        x = x + x
        y = y + y
        z = z + z
        if len(set(x_adjusted)) ==1: # adjust x
            # adjust nodes
            x_adjusted = x_adjusted + [xx + 1 for xx in x_adjusted]
            y_adjusted = y_adjusted + y_adjusted
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [(DimValue(name=x.name, value=x.value + 1), y, z) for x, y, z in verts_reduced]
        else: # adjust y
            # adjust nodes
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + [yx + 1 for yx in y_adjusted]
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [(x, DimValue(name=y.name, value=y.value + 1), z) for x, y, z in verts_reduced]

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
            verts_reduced += [(DimValue(name=x.name,value=x.value+1),y,z) for x,y,z in verts_reduced]
            # opposite face
            new_final_faces += [({DimValue(name=list(x_set)[0].name, value=list(x_set)[0].value + 1)}, y, z)
                                for x_set, y, z in final_faces]
            # side faces
            x_vs.add(DimValue(name=list(x_vs)[0].name, value=list(x_vs)[0].value + 1))
            for y_v in y_vs:
                new_final_faces += [(x_vs, {y_v}, z_vs) for _, y, z in final_faces]
            for z_v in z_vs:
                new_final_faces += [(x_vs, y_vs, {z_v}) for _, y, z in final_faces]
        if len(y_vs) == 1:
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + [yx + 1 for yx in y_adjusted]
            z_adjusted = z_adjusted + z_adjusted
            verts_reduced += [(x,DimValue(name=y.name,value=y.value+1),z) for x,y, z in verts_reduced]
            # opposite face
            new_final_faces += [(x, {DimValue(name=list(y_set)[0].name, value=list(y_set)[0].value + 1)}, z)
                                for x, y_set, z in final_faces]
            # side faces
            y_vs.add(DimValue(name=list(y_vs)[0].name, value=list(y_vs)[0].value + 1))
            for x_v in x_vs:
                new_final_faces += [({x_v}, y_vs, z_vs) for x, _, z in final_faces]
            for z_v in z_vs:
                new_final_faces += [(x_vs, y_vs, {z_v}) for x, _, z in final_faces]
        if len(z_vs) == 1:
            x_adjusted = x_adjusted + x_adjusted
            y_adjusted = y_adjusted + y_adjusted
            z_adjusted = z_adjusted + [zx + 1 for zx in z_adjusted]
            verts_reduced += [(x,y,DimValue(name=z.name,value=z.value+1)) for x,y,z in verts_reduced]
            # opposite face
            new_final_faces += [(x, y, {DimValue(name=list(z_set)[0].name, value=list(z_set)[0].value + 1)}) for x,y,z_set in final_faces]
            # side faces
            z_vs.add(DimValue(name=list(z_vs)[0].name, value=list(z_vs)[0].value+1))
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
    display_name = get_norm_display_name_interactive(norm_aspect_value)
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
    identifier_fixed = plotly_latex_fix(identifier)
    fig.add_trace(go.Scatter(x=[x.value+1], y=[y.value+1], name=identifier_fixed,
                             showlegend=showlegend, hovertemplate='%{text}',
                             legendgroup='case_'+identifier, text=[identifier_fixed],
                             mode='markers', marker=dict(color='#000000', size=6)),
                  row=row, col=col)
    return


def plot_case_interactive_3d(fig, case, dims, aspect_name):
    """
    Plots a case in a thre dimensional interactive figure.

    :param fig: Figure for plotting
    :param aspect_name: the aspect to plt (name)
    :param case: The case
    :param dims: The plotted dimensions
    """
    dim_x, dim_y, dim_z = dims
    coordinates = case.coordinates(aspect_name)
    fig.add_scatter3d(x=[coordinates[dim_x].value], y=[coordinates[dim_y].value],
                      z=[coordinates[dim_z].value], name=plotly_latex_fix(case.identifier),
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
    ticktexts = [plotly_latex_fix(val) for val in wrap_tick_texts(values, True)]
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


def wrap_tick_texts(text_list, interactive):
    """
    Since there is no auto-wrap for tick texts, this function wraps the tick texts. Each line is a latex expression.

    :param text_list: list of ticktexts to wrap
    :param interactive: whether it is for an interactive figure
    :return: the wrapped texts
    """
    new_text_list = []
    max_len = 8
    split_sign = '\\\\' if interactive else '$\n$'
    space_sign = '\\ ' if interactive else '$ $'
    for entry_id in range(len(text_list)):
        entry = text_list[entry_id]
        split = entry.split('\\ ')
        final_text = split[0]
        current_len = len(final_text)
        for i in range(1,len(split)):
            if current_len > max_len:
                final_text +=split_sign
                current_len = 0
            else:
                final_text += space_sign
            final_text += split[i]
            current_len += len(split[i])
        new_text_list.append(final_text)
    return new_text_list


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
        wrapped_tick_labels = wrap_tick_texts(scale, True)
        fig.update_xaxes(
            title_text=axis_label,
            ticktext=wrapped_tick_labels,
            autorange=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            automargin=True,
            nticks=len(scale),
            tickmode='array',
            tickangle=0,
            ticklabelstandoff=int(interactive_font_size*1.8),
            ticklabeloverflow='hide past div',
            range=[0, len(scale)+1],
            tickvals=[i+1 for i in range(len(scale))],
            row=row,
            col=col
        )
    else:
        fig.update_yaxes(
            title_text=axis_label,
            ticktext=wrap_tick_texts(scale, True),
            autorange=False,
            showgrid=False,
            tickwidth=1,
            ticklen=1,
            automargin=True,
            ticklabeloverflow='hide past div',
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
        ax.set_xticklabels(wrap_tick_texts(scale, False), fontsize=NormSystem.fontsize)
        ax.set_xlabel(label, loc="right", labelpad=20, fontsize=NormSystem.fontsize)
    else:
        ax.set_yticks([i + 0.5 for i in range(len(scale))])
        ax.set_yticklabels(wrap_tick_texts(scale, False), fontsize=NormSystem.fontsize)
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


def get_bounding_box_range(dim, norm_aspect_value, case, aspect_name):
    """
    Calculates start and end value for analogy box in one dimension for three-dimensional plotting.

    :param dim: The dimension to calculate bounding values for
    :param norm_aspect_value: The norm_aspect_value
    :param aspect_name: the aspect to plt (name)
    :param case: The case
    :return: start, end
    """
    values = sorted({case.coordinates(aspect_name)[dim], norm_aspect_value.start_values[dim], norm_aspect_value.end_values[dim]}, key=lambda x: x.value)
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


def get_analogy_display_name_interactive(norm_aspect_value, dim_one=False, case=None):
    """
    Generates the text to be displayed for an analogy in interactive figures.

    :param norm_aspect_value: The corresponding norm_aspect_value
    :param dim_one: Whether this is for a one dimensional figure
    :param case: The case this analogy belongs to
    :return: The generated display name
    """
    display_name = 'Analogy: ' + plotly_latex_fix(norm_aspect_value.identifier)
    if dim_one:
        display_name += plotly_latex_fix('_{' + norm_aspect_value.starttime + '}')
    if case is not None:
        display_name += ' to ' + plotly_latex_fix(case.identifier)
    return display_name


def plot_analogy_interactive_1d(fig, norm_aspect_value, row, col, outside, norm_index, x_vs, y_position, showlegend_outside, aspect):
    """
    Plots an analogy in a one dimensional interactive figure.

    :param fig: The figure for plotting
    :param norm_aspect_value: The corresponding norm_aspect_value
    :param row: row of the subplot
    :param col: column of the subplot
    :param aspect: The aspect to plot
    :param outside: List with truth values indicating whether the legend should be plotted (for updating)
    :param norm_index: Index of the norm_aspect_value in outside
    :param x_vs: x_min, x_start_line, x_max, x_end_line, scale
    :param y_position: the y position for plotting
    :param showlegend_outside: Whether to show the legend
    """
    x_min, x_start_line, x_max, x_end_line, scale = x_vs
    display_name = get_analogy_display_name_interactive(norm_aspect_value=norm_aspect_value, dim_one=True)
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
        line={'color': aspect.get_norm_color(norm_aspect_value), 'dash': 'dash'},
        mode='lines',
        name=display_name,
    ), row=row, col=col)


def plot_analogy_interactive_2d(fig, norm, x_vs, y_vs, showlegend, row, col, aspect):
    """
    Plots an analogy in two-dimensional interactive figures.

    :param fig: Figure for plotting
    :param norm: The corresponding norm_aspect_value
    :param x_vs: dim_x, x_min, x_max, x_scale
    :param aspect: the aspect to plot
    :param y_vs: dim_y, y_min, y_max, y_scale
    :param showlegend: Whether to show the legend
    :param row: row of the subplot
    :param col: column of the subplot
    """
    dim_x, x_min, x_max, x_scale = x_vs
    dim_y, y_min, y_max, y_scale = y_vs
    norm_color = aspect.get_norm_color(norm)
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


def plot_analogy_interactive_3d(figure, norm_aspect_value, dims, case, aspect):
    """
    Plots an analogy in three-dimensional interactive figures.

    :param figure: The figure for plotting
    :param norm_aspect_value: The corresponding norm_aspect_value
    :param aspect: The aspect to plot
    :param dims: The dimensions to plot
    :param case: The case this analogy belongs to
    """
    dim_x, dim_y, dim_z = dims
    norm_color = aspect.get_norm_color(norm_aspect_value)
    # get start and end coordinates for the bounding box
    bounding_values = get_bounding_box_range(dim=dim_x, norm_aspect_value=norm_aspect_value, case=case, aspect_name=aspect.aspect_name), \
                      get_bounding_box_range(dim=dim_y, norm_aspect_value=norm_aspect_value, case=case, aspect_name=aspect.aspect_name),\
                      get_bounding_box_range(dim=dim_z, norm_aspect_value=norm_aspect_value, case=case, aspect_name=aspect.aspect_name)

    analogy_shown = False
    for index, dim in [(0,dim_x), (1,dim_y), (2,dim_z)]:
        b_val = bounding_values[index]
        if b_val[0].value<norm_aspect_value.start_values[dim].value or b_val[1].value >norm_aspect_value.end_values[dim].value:
            analogy_shown = True

    if analogy_shown:
        vertices = [x for x in itertools.product(*bounding_values)]
        vert_map = {dim_x: {v[0].value:plotly_latex_fix(v[0].name) for v in vertices},
                    dim_y: {v[1].value:plotly_latex_fix(v[1].name) for v in vertices},
                    dim_z: {v[2].value:plotly_latex_fix(v[2].name) for v in vertices}}

        x_lines = list()
        y_lines = list()
        z_lines = list()
        x_lines_adjusted = list()
        y_lines_adjusted = list()
        z_lines_adjusted = list()

        bounding_values_adjusted = [[(x[0].value-0.5, x[0].value),(x[1].value+0.5, x[1].value)] for x in bounding_values]
        x_vs, y_vs, z_vs = bounding_values_adjusted

        for startpoint in [[x_vs[0], y_vs[0], z_vs[0]],[x_vs[1], y_vs[1], z_vs[1]]]:
            for change_id_one, change_id_two in [x for x in itertools.combinations([0,1,2], 2)]:
                # first triangle
                point_one = startpoint
                #change one
                point_two = swap_coordinate(to_swap=startpoint, swap_data=bounding_values_adjusted, swap_index = change_id_one)
                #change the other
                point_three = swap_coordinate(to_swap=startpoint, swap_data=bounding_values_adjusted, swap_index = change_id_two)
                #change both
                point_four = swap_coordinate(to_swap=point_three, swap_data=bounding_values_adjusted, swap_index = change_id_one)

                for point in [point_one, point_two, point_four, point_three, point_one]:
                    x_v, y_v, z_v = point
                    x_lines.append(x_v[1])
                    x_lines_adjusted.append(x_v[0])
                    y_lines.append(y_v[1])
                    y_lines_adjusted.append(y_v[0])
                    z_lines.append(z_v[1])
                    z_lines_adjusted.append(z_v[0])

                x_lines.append(None)
                y_lines.append(None)
                z_lines.append(None)
                x_lines_adjusted.append(None)
                y_lines_adjusted.append(None)
                z_lines_adjusted.append(None)

        texts = [None if xx is None else
                 '<i>' + dim_x + ': ' + vert_map[dim_x][xx] + '<br>'
                 + dim_y + ': ' + vert_map[dim_y][yx]
                 + '<br>' + dim_z + ': ' + vert_map[dim_z][zx] + '</i>'
                 for (xx, yx, zx) in zip(x_lines, y_lines, z_lines)]

        figure.add_scatter3d(
            x=x_lines_adjusted,
            y=y_lines_adjusted,
            z=z_lines_adjusted,
            hovertemplate='%{text}',
            text=texts,
            line={'dash': 'dash', 'color': norm_color},
            mode='lines',
            name=get_analogy_display_name_interactive(norm_aspect_value=norm_aspect_value, case=case),
            opacity=0.5,
        )


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
    for sign, start, end in [('_', '<sub>', '</sub>'),
                             ('\\\\', '<br>', ''),
                             ('^', '<sup>', '</sup>'),
                             ('\\', '&', ';')]:
        split_text = new_text.split(sign)
        new_text = split_text[0]
        for i in range(1,len(split_text)):
            curr_text = split_text[i]
            if len(new_text)>0:
                if sign == '\\' and curr_text[0] == ' ':
                    new_text += curr_text
                    continue
            if curr_text.startswith('{'):
                split_to_adjust = curr_text[1:].split('}')
                text_to_adjust = split_to_adjust[0]
                curr_text = '}'.join(split_to_adjust[1:])
            else:
                split_to_adjust = curr_text.split(' ')
                text_to_adjust = split_to_adjust[0]
                curr_text = ' '.join(split_to_adjust[1:])

            if '<' in text_to_adjust:
                var = text_to_adjust.split('<')
                text_to_adjust = var[0]
                curr_text = '<'+'<'.join(var[1:]) + curr_text

            if text_to_adjust.lower() in greek_letters and sign == '\\':
                capital = 'CAPITAL' if text_to_adjust[0].isupper() else 'SMALL'
                greek_letter_code = 'MATHEMATICAL ITALIC '+capital +' ' + text_to_adjust.upper()
                new_text = new_text + ud.lookup(greek_letter_code) + ' ' + curr_text
            else:
                new_text = new_text + start+ text_to_adjust + end +' '+ curr_text
    new_text = '<i>'+new_text+'</i>'
    return new_text


def setup_interactive_design(fig, dims_three=False):
    """
    Sets up general layout for interactive figures.

    :param fig: The figure
    :param dims_three: ehether this is the three-dimensional setting
    """

    font_size = interactive_font_size
    if dims_three:
        font_size -=4
    fig.update_layout(
        font_family="Times New Roman",
        font_size=font_size,
        title_font_family="Times New Roman",
        hoverlabel=dict(font=dict(family='Times New Roman')),
        template="plotly_white",
    )
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(
        margin=dict(
            pad=0
        )
    )


def create_dir(path):
    """
    Creates a directory if it doesn't exist yet.

    :param path: The path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_premise_identifier(norm_identifier, prem_id):
    """
    Creates the identifier for a premise with given id.

    :param norm_identifier: Correspnding norm
    :param prem_id: id
    :return: The created identifier
    """
    return r'$' + norm_identifier + '\\_' + str(prem_id) + '$'


class NormSystem:
    dim_o = 'O'
    dim_r = 'R'
    dim_s = 'S'
    dim_t = 'T'
    hierarchy = 'H'
    type_obl = 'obligation'
    type_perm = 'permission'
    type_prohib = 'prohibition'
    str_positive = 'positive'
    str_negative = 'negative'
    str_derogation = 'derogation'
    str_empowerment = 'empowerment'
    type_dero_pos = ' '.join([str_positive, str_derogation])
    type_empow_pos = ' '.join([str_positive, str_empowerment])
    type_dero_neg = ' '.join([str_negative, str_derogation])
    type_empow_neg = ' '.join([str_negative, str_empowerment])
    aspect_domain = 'ViNo'
    aspect_premise = 'ViVa'
    aspect_name_list = [aspect_premise, aspect_domain]
    initial_color_map = {type_perm:'#fcde78', type_obl:'#9cbad8', type_prohib:'#e66560',
                         type_dero_pos: '#9d4cfa', type_dero_neg: '#d6b4fd',
                         type_empow_pos: '#41891a', type_empow_neg: '#b1e495'}
    fontsize = 13
    dashed_linestyle = (7, (7, 4))
    linewidth = 2
    annotation_offset = 0.25
    dimensions = [dim_o, dim_r, dim_s, dim_t]

    def __init__(self, domain_scales, premise_scales):
        """
        Initializes the attributes.

        :param domain_scales scales for the domain aspect (object_scale, r_scale, subject_scale, time_scale)
        :param premise_scales scales for the premise aspect (object_scale, r_scale, subject_scale, time_scale)
        """
        self.norm_aspect_value_list = {key:[] for key in self.aspect_name_list}
        self.case_list = {key:[] for key in self.aspect_name_list}
        self.norm_list = []

        # possible hatches: https://stackoverflow.com/questions/14279344/how-can-i-add-textures-to-my-bars-and-wedges
        self.aspects = {NormSystem.aspect_premise:
                            NormAspect(aspect_name= NormSystem.aspect_premise, scales=premise_scales,
                                       norm_type_list=[NormType(name=NormSystem.type_empow_pos, color=self.initial_color_map[self.type_empow_pos], hatch='//'),
                                                       NormType(name=NormSystem.type_empow_neg, color=self.initial_color_map[self.type_empow_neg], hatch='*'),
                                                       NormType(name=NormSystem.type_dero_pos, color=self.initial_color_map[self.type_dero_pos], hatch='\\\\'),
                                                       NormType(name=NormSystem.type_dero_neg, color=self.initial_color_map[self.type_dero_neg], hatch='o')]),
                        NormSystem.aspect_domain:
                            NormAspect(aspect_name = NormSystem.aspect_domain, scales=domain_scales,
                                norm_type_list=
                                 [NormType(name=NormSystem.type_obl, color=self.initial_color_map[self.type_obl], hatch='*'),
                                  NormType(name=NormSystem.type_perm, color=self.initial_color_map[self.type_perm], hatch='//',
                                           contrary_type_name=NormSystem.type_prohib),
                                  NormType(name=NormSystem.type_prohib, color=self.initial_color_map[self.type_prohib], hatch='\\\\',
                                           contrary_type_name=NormSystem.type_perm),
                                  ])}

    def reset_colors(self, aspect_name):
        """
        Resets the colors of the norms to the original color setting.
        """
        aspect = self.aspects[aspect_name]
        aspect.colors = {key:self.initial_color_map[key] for key in aspect.type_names}

    def change_color(self, norm_type, color_hex, aspect_name):
        """
        Changes the color for the given normative type to the given hex code

        :param norm_type: One of the three norm_aspect_value types
        :param aspect_name: aspect to change the color for
        :param color_hex: A hexadecimal code for a color
        """
        aspect = self.aspects[aspect_name]
        aspect.colors[norm_type] = color_hex

    def add_norm(self, norm):
        """
        Method for adding a domain norm to the system.

        :param norm: norm to add
        """
        # check name
        name_exists = False
        for existing_norm in self.norm_list:
            if (existing_norm.domain_aspect().identifier  ==
                    norm.domain_aspect().identifier):
                name_exists = True
        if not name_exists:
            self.norm_list.append(norm)
            self.norm_aspect_value_list[NormSystem.aspect_domain].append(norm.norm_aspect_values[NormSystem.aspect_domain][0])

    def add_norm_premise_aspect(self, norm_identifier, premise_aspect):
        """
        Adds a premise aspect to an existing norm.

        :param norm_identifier: The norm to add the aspect to
        :param premise_aspect: The premise aspect to add
        :return: An abort reason if there was an error
        """
        abort_reason = ''
        norm = self.get_norm(identifier=norm_identifier)

        if norm is not None:
            other_premises = norm.norm_aspect_values[NormSystem.aspect_premise]
            prem_id = 1
            identifier = get_premise_identifier(norm_identifier, str(prem_id))
            value_names = set(norm_aspect_value.identifier for norm_aspect_value in other_premises)
            while identifier in value_names:
                prem_id += 1
                identifier = get_premise_identifier(norm_identifier, str(prem_id))
            premise_aspect.identifier = identifier

            if len(other_premises) > 0: # there are already other premises
                first_prem = other_premises[0]
                norm_type = NormSystem.str_empowerment if (NormSystem.str_empowerment
                                                                 in first_prem.norm_type)\
                                                            else NormSystem.str_derogation
                if norm_type in premise_aspect.norm_type: # all are of the same general type (as the first)
                    if NormSystem.str_positive in premise_aspect.norm_type: # add if positive
                        norm.add_premise_aspect(premise_aspect=premise_aspect)
                        self.norm_aspect_value_list[NormSystem.aspect_premise].append(premise_aspect)
                    else:
                        subset = False
                        positive_prems = [v for v in other_premises if NormSystem.str_positive in v.norm_type]
                        for pos_prem in positive_prems:
                            cur_subset = True
                            for dim in NormSystem.dimensions:
                                if pos_prem.start_values[dim].value>premise_aspect.start_values[dim].value:
                                    cur_subset = False
                                if pos_prem.end_values[dim].value<premise_aspect.end_values[dim].value:
                                    cur_subset = False
                            if cur_subset:
                                subset=True
                                break
                        if subset: # make sure it is a subset of one premise
                            norm.add_premise_aspect(premise_aspect=premise_aspect)
                            self.norm_aspect_value_list[NormSystem.aspect_premise].append(premise_aspect)
                        else:
                            abort_reason = 'No negative '+NormSystem.aspect_premise+' norm may contain values not contained in a positive '+NormSystem.aspect_premise+' norm.'
                else:
                    abort_reason = ('All ' + NormSystem.aspect_premise + ' norms of a '+NormSystem.aspect_domain
                                    +' norm must be either '+NormSystem.str_derogation+' or '+NormSystem.str_empowerment+'.')
            else: # first premise aspect
                if NormSystem.str_positive in premise_aspect.norm_type:
                    norm.add_premise_aspect(premise_aspect=premise_aspect)
                    self.norm_aspect_value_list[NormSystem.aspect_premise].append(premise_aspect)
                else:
                    abort_reason = 'First '+NormSystem.aspect_premise+' norm must be '+NormSystem.str_positive+'.'
        else:
            abort_reason = 'No corresponding norm found.'
        return abort_reason

    def delete_norm_premise_aspect(self, premise_aspect):
        """
        Deletes a premis aspect from an existing norm

        :param premise_aspect: The aspect to delte
        :return: An abort reason if there was an error
        """
        norm = self.get_norm('\\_'.join(premise_aspect.identifier.replace('$','').split('\\_')[:-1]))
        abort_reason = ''
        if norm is not None:
            premises = norm.norm_aspect_values[NormSystem.aspect_premise]
            premises.remove(premise_aspect)
        else:
            abort_reason = 'No correspnding '+NormSystem.aspect_domain+' norm found.'
        return abort_reason

    def get_norm(self, identifier):
        """
        Finds a norm in the norm list

        :param identifier: identifier of the domain aspect
        :return: the found Norm or None
        """
        for norm in self.norm_list:
            if norm.domain_aspect().identifier == identifier:
                return norm
        return None

    def add_case(self, domain_aspect, premise_aspect, identifier=None):
        """
        Adds a case to the norm_aspect_value system.

        :param domain_aspect: Information on the domain aspect
        :param premise_aspect: Information on the premise aspect
        :param identifier: Name of the norm_aspect_value. If non is given, it will be C_x with x the current count of cases
                            in the NormSystem
        """
        if identifier is None:
            case_id = 1
            identifier = r'$\mathfrak{C}_{' + str(case_id) + '}$'
            case_names = set([case.identifier for case in self.case_list[NormSystem.aspect_domain]]+[case.identifier for case in self.case_list[NormSystem.aspect_premise]])
            while identifier in case_names:
                case_id += 1
                identifier = r'$\mathfrak{C}_{' + str(case_id) + '}$'
        case_aspect_list = []
        if premise_aspect is not None:
            case_aspect_list.append(premise_aspect)
        case_aspect_list.append(domain_aspect)
        case = Case(identifier=identifier, case_aspect_list=case_aspect_list)
        for aspect in case_aspect_list:
            self.case_list[aspect.aspect_name].append(case)

    def delete_domain_norm(self, norm):
        """
        Deletes a norm_aspect_value from the list of norms.

        :param norm: The norm_aspect_value to delete.
        """
        for aspect_name in self.aspect_name_list:
            for aspect_value in norm.norm_aspect_values[aspect_name]:
                self.norm_aspect_value_list[aspect_name].remove(aspect_value)
        self.norm_list.remove(norm)

    def delete_case(self, case):
        """
        Deletes a case from the list of cases.

        :param case: The case to delete.
        """
        for aspect_name in self.aspect_name_list:
            if case in self.case_list[aspect_name]:
                self.case_list[aspect_name].remove(case)

    def get_relevant_norm_aspect_values(self, aspect_name):
        """
        Is called when a figure in a NormSystem should be plotted. Selects all normas relevant to the cases.
        These are all norms which subsume a case and in case that a case is not subsumed by any norm_aspect_value, the k closest
        norms are returned for that norm_aspect_value.

        :return: a set of all relevant norms.
        """
        for norm_aspect_value in self.norm_aspect_value_list[aspect_name]:
            norm_aspect_value.reset_outside_cases()
        result_list = set()
        for case in self.case_list[aspect_name]:
            fitting_norm_aspect_values = []
            if aspect_name == NormSystem.aspect_premise: # show all subsuming ones
                for norm_aspect_value in self.norm_aspect_value_list[aspect_name]:
                    fitting_norm_aspect_values.append(norm_aspect_value)
                    if not norm_aspect_value.subsumes(case):
                        norm_aspect_value.add_outside_case(case)
            else: # domain
                # else: only show if smallest subsuming viva is ... or no subsuming viva
                for norm in self.norm_list:
                    subsuming_premises = [premise for premise in norm.norm_aspect_values[NormSystem.aspect_premise] if premise.subsumes(case)]
                    if len(subsuming_premises)>0:
                        subsuming_premises.sort(key=lambda x: sum(
                            [x.end_values[dim].value - x.start_values[dim].value for dim in NormSystem.dimensions]))
                        smallest_subsumin = subsuming_premises[0]
                        if smallest_subsumin.norm_type in [NormSystem.type_empow_pos, NormSystem.type_dero_neg]:
                            fitting_norm_aspect_values.append(norm.domain_aspect())
                            if not norm.domain_aspect().subsumes(case):
                                norm.domain_aspect().add_outside_case(case)
                    else: # no subsuming premise at all
                        fitting_norm_aspect_values.append(norm.domain_aspect())
                        if not norm.domain_aspect().subsumes(case):
                            norm.domain_aspect().add_outside_case(case)
            result_list = result_list.union(fitting_norm_aspect_values)

        if aspect_name == NormSystem.aspect_domain:
            result_list = ([v for v in result_list if v.norm_type==NormSystem.type_obl]+
                           [v for v in result_list if v.norm_type==NormSystem.type_prohib]+
                           [v for v in result_list if v.norm_type==NormSystem.type_perm])

        return result_list

    def get_relevant_norms_sorted_by_hierarchy(self, aspect_name):
        """
        Returns the relevant norm_aspect_value and sorts them by hierarchy (for one dimensional plots)

        :return: A list of len(hierarchy_scale) with each position containing a list of norms in that hierarchy level
        """
        norm_aspect_values = self.get_relevant_norm_aspect_values(aspect_name)

        sorted_norm_aspect_values = []
        for h_name in self.aspects[aspect_name].scales[NormSystem.hierarchy]:
            h_norms = [norm_aspect_v for norm_aspect_v in norm_aspect_values if norm_aspect_v.hierarchy.name == h_name]
            sorted_norm_aspect_values.append(h_norms)

        return sorted_norm_aspect_values

    def get_position(self, value, dimension, aspect_name):
        """
        Gets the mathematical value for the given literal value on a scale.

        :param value: Value to get mathematical value for.
        :param dimension: dimension from NormSystem.dimensions
        :param aspect_name: aspect to get position to
        :return:
        """
        scale = self.aspects[aspect_name].scales[dimension]
        return scale.index(value)

    def draw_one_dim_subplot(self, ax, dim, detailed, aspect_name):
        """
        Draws one subplot for a one dimensional plot.

        :param ax: axes of the plot
        :param dim: x-axis from NormSystem.dimensions
        :param aspect_name: the aspect for plotting
        :param detailed: whether to draw the detailed version
        """
        scale = self.aspects[aspect_name].scales[dim]
        aspect = self.aspects[aspect_name]

        # cases
        annotations = {}
        for case in self.case_list[aspect_name]:
            annotations = plot_case(ax, x=case.coordinates(aspect_name)[dim].value, y=-1,
                                    annotations=annotations, identifier=case.identifier)
        annotate_cases(ax, annotations)

        # norms
        relevant_norms_aspect_values = self.get_relevant_norms_sorted_by_hierarchy(aspect_name)
        for norms_aspect_vales_per_hierarchy in relevant_norms_aspect_values:
            num_norms = len(norms_aspect_vales_per_hierarchy)
            offset = 1 / (num_norms + 1)
            for i in range(len(norms_aspect_vales_per_hierarchy)):
                current_offset = offset * (i + 1)

                norm_aspect_value = norms_aspect_vales_per_hierarchy[i]
                end_point = norm_aspect_value.end_values[dim]
                start_point = norm_aspect_value.start_values[dim]
                norm_color = aspect.get_norm_color(norm_aspect_value)
                y_position = norm_aspect_value.hierarchy.value + current_offset
                ax.hlines(y=y_position, xmax=end_point.value + 1, xmin=start_point.value,
                          color=norm_color,
                          linewidth=NormSystem.linewidth)

                # dashed lines from axes
                dashes_lines_x_points = [start_point.value, end_point.value + 1]

                x_start_line = x_min = start_point.value
                x_end_line = x_max = end_point.value + 1
                # outside
                if len(norm_aspect_value.outside_cases) > 0:
                    for case in norm_aspect_value.outside_cases:
                        x_case = case.coordinates(aspect_name)[dim].value
                        x_max = max(x_case + 1, x_max)
                        x_min = min(x_case, x_min)
                    if x_min < x_start_line:
                        dashes_lines_x_points.append(x_min)
                    if x_max > x_end_line:
                        dashes_lines_x_points.append(x_max)

                    if norm_aspect_value.norm_type in aspect.contrary_map:
                        contrary_color = aspect.get_contrary_norm_color(norm_aspect_value)
                        add_contrary_arrow(ax, start=(x_start_line, y_position - (0.3 * offset)),
                                           length=(x_min - x_start_line, 0),
                                           color=contrary_color)
                        add_contrary_arrow(ax, start=(x_end_line, y_position - (0.3 * offset)),
                                           length=(x_max - x_end_line, 0),
                                           color=contrary_color)

                    add_analogy_h_line(ax=ax, x_start=x_start_line, x_end=x_min, color=norm_color, y=y_position)
                    add_analogy_h_line(ax=ax, x_start=x_end_line, x_end=x_max, color=norm_color, y=y_position)

                annotate_norm(ax=ax, norm_aspect_value=norm_aspect_value, x=x_max, y=y_position, dim_one=True)
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
        prepare_hierarchy_axis(scale=self.aspects[aspect_name].scales[NormSystem.hierarchy], ax=ax)

    def draw_one_dim_subplot_interactive(self, fig, dim, row, col, outside, aspect_name):
        """
        Draws one subplot for a one dimensional plot (interactive).

        :param fig: figure of the plot
        :param dim: x-axis from NormSystem.dimensions
        :param row: row of the subplot
        :param col: column of the subplot
        :param aspect_name: the aspect to draw the plot for
        :param outside: List with truth values indicating whether the legend should be plotted (for updating)
        :return: The updated outside
        """
        scale = self.aspects[aspect_name].scales[dim]
        showlegend = (col+row) == 2 # only show legend for first plot
        setup_interactive_design(fig)
        aspect = self.aspects[aspect_name]

        # cases
        for case in self.case_list[aspect_name]:
            plot_case_interactive(fig, x=case.coordinates(aspect_name)[dim], y=DimValue(name='',value=-0.65), identifier=case.identifier, row=row, col=col, showlegend=showlegend)

        # norms
        relevant_norm_aspect_values_per_hierarchy = self.get_relevant_norms_sorted_by_hierarchy(aspect_name)
        for norm_aspect_values in relevant_norm_aspect_values_per_hierarchy:
            num_norms = len(norm_aspect_values)
            offset = 1 / (num_norms + 1)

            for i in range(len(norm_aspect_values)):
                current_offset = offset * (i + 1)
                norm_aspect_value = norm_aspect_values[i]
                y_position = norm_aspect_value.hierarchy.value + current_offset + 0.5

                plot_norm_interactive_1d(fig=fig, norm_aspect_value=norm_aspect_value, y_position=y_position, dim=dim, showlegend=showlegend, row=row, col=col, aspect=aspect)

                x_start_line = x_min = norm_aspect_value.start_values[dim].value
                x_end_line = x_max = norm_aspect_value.end_values[dim].value + 1
                # outside
                if len(norm_aspect_value.outside_cases) > 0:
                    showlegend_outside = showlegend | (not outside[self.norm_aspect_value_list[aspect_name].index(norm_aspect_value)])  # show analogy etc. legend only if it wasn't plotted before
                    for case in norm_aspect_value.outside_cases:
                        x_case = case.coordinates(aspect_name)[dim].value
                        x_max = max(x_case + 1, x_max)
                        x_min = min(x_case, x_min)

                    # contrary arrows
                    if norm_aspect_value.norm_type in aspect.contrary_map:
                        plot_contrary_arrows_interactive_1d(fig=fig, row=row, col=col, outside=outside,
                                                            x_vs=(x_min, x_max, x_start_line, x_end_line, scale),
                                                            norm=norm_aspect_value, showlegend_outside=showlegend_outside,
                                                            norm_index=self.norm_aspect_value_list[aspect_name].index(norm_aspect_value),
                                                            y_pos_with_offset=y_position - (0.3 * offset),
                                                            aspect=aspect)

                    # analogy
                    plot_analogy_interactive_1d(fig=fig, norm_aspect_value=norm_aspect_value, row=row, col=col, outside=outside,
                                                norm_index= self.norm_aspect_value_list[aspect_name].index(norm_aspect_value),
                                                x_vs=(x_min, x_start_line, x_max, x_end_line, scale),
                                                y_position=y_position, showlegend_outside=showlegend_outside, aspect=aspect)

        # axis
        prepare_one_axis_interactive(scale=scale, axis_label=dim, fig=fig, x_axis=True, row=row, col=col)
        prepare_hierarchy_axis_interactive(scale=self.aspects[aspect_name].scales[NormSystem.hierarchy], fig=fig, row=row, col=col)
        return outside

    def draw_two_dim_subplot_interactive(self, fig, dim_x, dim_y, row, col, outside, aspect_name):
        """
        Draws one subplot for a two-dimensional plot.

        :param fig: figure of the plot
        :param dim_x: x-axis from NormSystem.dimensions
        :param dim_y: y-axis from NormSystem.dimensions
        :param row: row of the subplot
        :param col: column of the subplot
        :param aspect_name: the aspect to draw the plot for
        :param outside: List with truth values indicating whether the legend should be plotted (for updating)
        """
        showlegend = (col + row) == 2  # only show legend for first plot
        setup_interactive_design(fig)
        aspect = self.aspects[aspect_name]

        # norms
        for norm_aspect_value in self.get_relevant_norm_aspect_values(aspect_name):
            plot_norm_interactive_2d(fig=fig, norm_aspect_value=norm_aspect_value, dim_x=dim_x, dim_y=dim_y, showlegend=showlegend, col=col, row=row, aspect=aspect)

            if len(norm_aspect_value.outside_cases) > 0:
                showlegend_outside = showlegend | (not outside[self.norm_aspect_value_list[aspect_name].index(norm_aspect_value)])  # show analogy etc. legend only if it wasn't plotted before

                x_start_rect = x_min = norm_aspect_value.start_values[dim_x].value
                y_start_rect = y_min = norm_aspect_value.start_values[dim_y].value
                y_end_rect = y_max = norm_aspect_value.end_values[dim_y].value + 1
                x_end_rect = x_max = norm_aspect_value.end_values[dim_x].value + 1
                for case in norm_aspect_value.outside_cases:
                    y_case = case.coordinates(aspect_name)[dim_y].value
                    x_case = case.coordinates(aspect_name)[dim_x].value
                    y_max = max(y_case + 1, y_max)
                    y_min = min(y_case, y_min)
                    x_max = max(x_case + 1, x_max)
                    x_min = min(x_case, x_min)

                # contrary arrows
                if norm_aspect_value.norm_type in aspect.contrary_map:
                    plot_contrary_arrows_interactive_2d(fig=fig, col=col, row=row, norm=norm_aspect_value, outside=outside,
                                                        x_vs = (x_min, x_max, x_start_rect, x_end_rect),
                                                        y_vs = (y_min, y_max, y_start_rect, y_end_rect),
                                                        norm_index=self.norm_aspect_value_list[aspect_name].index(norm_aspect_value), showlegend_outside=showlegend_outside, aspect=aspect)

                if x_min != x_start_rect or x_max != x_end_rect or y_min != y_start_rect or y_max != y_end_rect:
                    # analogy
                    plot_analogy_interactive_2d(fig=fig, row=row, col=col, showlegend=showlegend,
                                                x_vs=(dim_x, x_min+0.5,x_max+0.5, self.aspects[aspect_name].scales[dim_x]),
                                                y_vs=(dim_y, y_min+0.5,y_max+0.5, self.aspects[aspect_name].scales[dim_y]),
                                                norm=norm_aspect_value, aspect=aspect)

        # cases, down here to plot above the rest
        for case in self.case_list[aspect_name]:
            plot_case_interactive(fig, x=case.coordinates(aspect_name)[dim_x], y=case.coordinates(aspect_name)[dim_y], identifier=case.identifier, row=row, col=col,
                                  showlegend=showlegend)

        prepare_one_axis_interactive(scale=self.aspects[aspect_name].scales[dim_x], axis_label=dim_x, fig=fig, x_axis=True, row=row, col=col)
        prepare_one_axis_interactive(scale=self.aspects[aspect_name].scales[dim_y], axis_label=dim_y, fig=fig, x_axis=False, row=row, col=col)
        return outside

    def draw_two_dim_subplot(self, ax, dim_x, dim_y, detailed, aspect_name):
        """
        Draws one subplot for a two-dimensional plot.

        :param ax: axes of the plot
        :param dim_x: x-axis from NormSystem.dimensions
        :param dim_y: y-axis from NormSystem.dimensions
        :param detailed: whether to draw the detailed version
        :param aspect_name: aspect to draw the plot for
        """
        aspect = self.aspects[aspect_name]
        # cases
        annotations = {}
        for case in self.case_list[aspect_name]:
            annotations = plot_case(ax, x=case.coordinates(aspect_name)[dim_x].value, y=case.coordinates(aspect_name)[dim_y].value,
                                    annotations=annotations, identifier=case.identifier)
        annotate_cases(ax, annotations)

        # norms
        norm_annotations = {}
        for norm_aspet_value in self.get_relevant_norm_aspect_values(aspect_name):
            norm_color = aspect.get_norm_color(norm_aspet_value)
            start = {dim_x:norm_aspet_value.start_values[dim_x],
                            dim_y:norm_aspet_value.start_values[dim_y]}
            end = {dim_x:norm_aspet_value.end_values[dim_x],
                          dim_y:norm_aspet_value.end_values[dim_y]}

            ax.add_patch(Rectangle((start[dim_x].value, start[dim_y].value),
                                   end[dim_x].value - start[dim_x].value + 1,
                                   end[dim_y].value - start[dim_y].value + 1,
                                   facecolor='none',
                                   hatch=aspect.hatches[norm_aspet_value.norm_type], linewidth=NormSystem.linewidth,
                                   edgecolor=norm_color))
            x, y = end[dim_x].value + 1, start[dim_y].value + (end[dim_y].value + 1 -
                                                                                 start[dim_y].value) / 2
            if (x, y) not in norm_annotations:
                norm_annotations[(x, y)] = [norm_aspet_value]
            else:
                norm_annotations[(x, y)].append(norm_aspet_value)

            # dashed lines from axes
            dashed_y_start = start[dim_y].value
            dashed_x_start = start[dim_x].value
            dashes_lines_x_points = [start[dim_x].value, end[dim_x].value + 1]
            dashes_lines_y_points = [start[dim_y].value, end[dim_y].value + 1]

            if len(norm_aspet_value.outside_cases) > 0:
                x_start_rect = x_min = start[dim_x].value
                y_start_rect = y_min = start[dim_y].value
                y_end_rect = y_max = end[dim_y].value + 1
                x_end_rect = x_max = end[dim_x].value + 1
                for case in norm_aspet_value.outside_cases:
                    y_case = case.coordinates(aspect_name)[dim_y].value
                    x_case = case.coordinates(aspect_name)[dim_x].value
                    y_max = max(y_case + 1, y_max)
                    y_min = min(y_case, y_min)
                    x_max = max(x_case + 1, x_max)
                    x_min = min(x_case, x_min)
                # contrary arrows
                if norm_aspet_value.norm_type in aspect.contrary_map:
                    contrary_color = aspect.get_contrary_norm_color(norm_aspet_value)
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
                                               color=contrary_color)
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
                                               color=contrary_color)
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
                                               color=contrary_color)
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
                                               color=contrary_color)

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
                annotate_norm(ax=ax, norm_aspect_value=norms[i], x=x, y=y + (i * 0.25))

        # arrows on axes
        ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False)
        ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(NormSystem.linewidth)
        prepare_one_axis(scale=self.aspects[aspect_name].scales[dim_x], axis_label=dim_x, ax=ax, x_axis=True)
        prepare_one_axis(scale=self.aspects[aspect_name].scales[dim_y], axis_label=dim_y, ax=ax, x_axis=False)

    def draw_three_dim_subplot(self, fig, dim_to_exclude, aspect_name):
        """
        Draws a three-dimensional plot.

        :param fig: The figure
        :param aspect_name: The aspect to draw the figure for
        :param dim_to_exclude: The dimension not to display
        """
        dims = NormSystem.dimensions.copy()
        dims.remove(dim_to_exclude)
        dim_x, dim_y, dim_z = dims[0], dims[1], dims[2]
        setup_interactive_design(fig, dims_three=True)

        aspect = self.aspects[aspect_name]
        prepare_3d_axes(fig=fig, x_vs=(dim_x, self.aspects[aspect_name].scales[dim_x]),
                        y_vs=(dim_y, self.aspects[aspect_name].scales[dim_y]), z_vs=(dim_z, self.aspects[aspect_name].scales[dim_z]))

        # cases
        for case in self.case_list[aspect_name]:
            plot_case_interactive_3d(fig=fig, case=case, dims=(dim_x, dim_y, dim_z), aspect_name=aspect_name)

        # norms
        for norm_aspect_value in self.get_relevant_norm_aspect_values(aspect_name):
            plot_norm_interactive_3d(figure=fig, norm_aspect_value=norm_aspect_value, dims=dims, aspect=aspect)

            # analogies
            for case in norm_aspect_value.outside_cases:
                plot_analogy_interactive_3d(figure=fig, norm_aspect_value=norm_aspect_value, dims=dims, case=case, aspect=aspect)

                # contrary arrows
                if norm_aspect_value.norm_type in aspect.contrary_map:
                    plot_contrary_arrows_interactive_3d(figure=fig, norm_aspect_value=norm_aspect_value, dims=dims, aspect=aspect)

    def draw_dims_one_all(self, aspect_name, display_type=display_classic_detailed, figure=None, saveidentifier=None):
        """
        Coordinates drawing figures for one dimenions each.

        :param saveidentifier: an identifier to include in the name of the saved file.
        :param figure: figure to use for plotting
        :param aspect_name: the aspect to plot for
        :param display_type:  Indicates which version to draw for the figures.
        """
        if display_type != display_interactive:
            if saveidentifier is not None:
                fig = plt.figure(figsize=(15, 10))
            else:
                fig = figure

            if fig is not None:
                matplotlib.rcParams['mathtext.fontset'] = 'cm'
                ax1 = fig.add_subplot(221)  # Plot with: 4 rows, 1 column
                self.draw_one_dim_subplot(ax1, NormSystem.dim_o, display_type == display_classic_detailed, aspect_name=aspect_name)
                ax2 = fig.add_subplot(222)
                self.draw_one_dim_subplot(ax2, NormSystem.dim_r, display_type == display_classic_detailed, aspect_name=aspect_name)
                ax3 = fig.add_subplot(223)
                self.draw_one_dim_subplot(ax3, NormSystem.dim_s, display_type == display_classic_detailed, aspect_name=aspect_name)
                ax4 = fig.add_subplot(224)
                self.draw_one_dim_subplot(ax4, NormSystem.dim_t, display_type == display_classic_detailed, aspect_name=aspect_name)

                fig.tight_layout()
            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_'+aspect_name+'_dims_one.png'
                plt.savefig(savename)
                plt.close()
        else:
            fig = make_subplots(rows=2, cols=2)
            outside = [False]*len(self.norm_aspect_value_list[aspect_name])
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_o, row=1, col=1, outside=outside, aspect_name=aspect_name)
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_r, row=1, col=2, outside=outside, aspect_name=aspect_name)
            outside = self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_s, row=2, col=1, outside=outside, aspect_name=aspect_name)
            self.draw_one_dim_subplot_interactive(fig, NormSystem.dim_t, row=2, col=2, outside=outside, aspect_name=aspect_name)

            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_'+aspect_name+'_dims_one.html'
                fig.write_html(savename, include_mathjax='cdn')
        return fig

    def draw_dims_one(self, display_type, x_dim, fig, aspect_name):
        """
        Coordinates drawing figures for one dimenions each.

        :param x_dim: dimension to display on x-axis
        :param fig: figure to use for plotting
        :param aspect_name: the aspect for plotting
        :param display_type: How to draw the figures.
        """
        if display_type != display_interactive:
            matplotlib.rcParams['mathtext.fontset'] = 'cm'
            ax1 = fig.add_subplot(111)
            self.draw_one_dim_subplot(ax1, x_dim, display_type==display_classic_detailed, aspect_name=aspect_name)
            fig.tight_layout()
        else:
            fig = make_subplots(rows=1, cols=1)
            outside = [False]*len(self.norm_aspect_value_list[aspect_name])
            self.draw_one_dim_subplot_interactive(fig, x_dim, row=1, col=1, outside=outside, aspect_name=aspect_name)
        return fig

    def draw_dims_two(self, dim_x, dim_y, display_type, fig, aspect_name):
        """
        Coordinates drawing figures for two dimenions each.

        :param dim_x: x-axis (from NormSystem.dimensions)
        :param dim_y: y-axis (from NormSystem.dimensions)
        :param fig: figure for plotting
        :param aspect_name: The aspect to draw the diagram for
        :param display_type: How to display the figure.
        """
        if display_type != display_interactive:
            matplotlib.rcParams['mathtext.fontset'] = 'cm'
            ax1 = fig.add_subplot(111)  # Plot with: 1 row, 2 column, first subplot.
            self.draw_two_dim_subplot(ax1, dim_x, dim_y, display_type==display_classic_detailed, aspect_name)
            fig.tight_layout()
        else:
            fig = make_subplots(rows=1, cols=1)
            outside = [False]*len(self.norm_aspect_value_list[aspect_name])
            self.draw_two_dim_subplot_interactive(fig, dim_x, dim_y, row=1, col=1, outside=outside, aspect_name=aspect_name)
        return fig


    def draw_dims_two_all(self, dims_one, dims_two, aspect_name, display_type=display_classic_detailed, saveidentifier=None, figure=None):
        """
        Coordinates drawing figures for two dimenions each.

        :param dims_one: (dim_x_str, dim_y_str) for first plot (from NormSystem.dimensions)
        :param dims_two: (dim_x_str, dim_y_str) for second plot (from NormSystem.dimensions)
        :param figure: figure for plotting
        :param saveidentifier: an identifier to include in the name of the saved file.
        :param display_type: Indicator for the version to plot
        :param aspect_name: The aspect to draw the diagram for
        """
        if display_type != display_interactive:
            if saveidentifier is not None:
                fig = plt.figure(figsize=(20, 10))
            else:
                fig = figure

            if fig is not None:
                matplotlib.rcParams['mathtext.fontset'] = 'cm'
                ax1 = fig.add_subplot(121)  # Plot with: 1 row, 2 column, first subplot.
                self.draw_two_dim_subplot(ax1, dims_one[0], dims_one[1], display_type == display_classic_detailed, aspect_name=aspect_name)
                ax2 = fig.add_subplot(122)  # Plot with: 1 row, 2 column, second subplot.
                self.draw_two_dim_subplot(ax2, dims_two[0], dims_two[1], display_type == display_classic_detailed, aspect_name=aspect_name)
                fig.tight_layout()

            if saveidentifier is not None:
                path = display_type + '/'
                create_dir(path)
                savename = path + saveidentifier + '_'+aspect_name+'_dims_two.png'
                plt.savefig(savename)
                plt.close()
        else:
            fig = make_subplots(rows=1, cols=2,
                                horizontal_spacing=0.15,
                                )
            outside = [False]*len(self.norm_aspect_value_list[aspect_name])
            outside = self.draw_two_dim_subplot_interactive(fig, dims_one[0], dims_one[1], row=1, col=1, outside=outside, aspect_name=aspect_name)
            self.draw_two_dim_subplot_interactive(fig, dims_two[0], dims_two[1], row=1, col=2, outside=outside, aspect_name=aspect_name)

            if saveidentifier is not None:
                path = display_type+'/'
                create_dir(path)
                savename = path + saveidentifier + '_'+aspect_name+'_dims_two.html'
                fig.write_html(savename, include_mathjax='cdn')
        return fig

    def draw_dims_three(self, dim_to_exclude, aspect_name, saveidentifier=None):
        """
        Wrapper method for plotting a three-dimensional diagram and saving it.

        :param saveidentifier: Identifier for saving
        :param dim_to_exclude: The dimension not to display
        :param aspect_name: the aspect to draw the figure for
        """

        fig = make_subplots(rows=1, cols=1)
        self.draw_three_dim_subplot(fig, dim_to_exclude, aspect_name)
        if saveidentifier is not None:
            path = display_interactive + '/'
            create_dir(path)
            savename = path + saveidentifier + '_' + aspect_name + '_dims_three.html'
            fig.write_html(savename, include_mathjax='cdn')
        return fig

    def reset(self):
        """
        Removes all norms and cases from the NormSystem.
        """
        self.norm_aspect_value_list = {key: [] for key in self.aspect_name_list}
        self.case_list = {key: [] for key in self.aspect_name_list}
        self.norm_list = []


class Case:
    """
    Class for one case.
    """

    def __init__(self, identifier, case_aspect_list):
        """
        Initializes the attributes.

        :param identifier: Name / Identifier of the case
        """
        self.case_aspect_values = {case_aspect.aspect_name: case_aspect for case_aspect in case_aspect_list}
        self.identifier = identifier

    def coordinates(self, aspect_name):
        return self.case_aspect_values[aspect_name].coordinates

class CaseAspectValues:
    """
    Holds the values for a case for a specific aspect
    """
    def __init__(self, o_val, r_val, s_val, t_val, aspect_name):
        """
        Initializes the attributes.

        :param o_val: coordinate on dimension o
        :param r_val: coordinate on dimension r
        :param s_val: coordinate on dimension s
        :param t_val: coordinate on dimension t
        """
        self.aspect_name=aspect_name
        self.coordinates = {NormSystem.dim_o: o_val, NormSystem.dim_r: r_val,
                            NormSystem.dim_s: s_val, NormSystem.dim_t: t_val}

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
        res = np.append(res, [vertice_with_names[i].value])
    return res


def is_value_in_range(value, val_range):
    """
    Checks whether a value is in a given range.

    :param value: A mathematical value (number)
    :param val_range: A set of tuples (ordered) which define the range. Each tuple includes the literal coordinate
                      name and the mathematical value of the coordinate. Range is inclusive at the end.
    :return: True, if the given value is in the range, False otherwise.
    """
    range_vals = sorted([val.value for val in val_range])
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

    def __init__(self, domain_aspect):
        """
        Norms are hyperrectangles https://de.wikipedia.org/wiki/Hyperrechteck
        (https://de.wikipedia.org/wiki/Hyperw%C3%BCrfel for visual reference on faces and nodes).

        :param domain_aspect: Information on the domain aspect
        """
        self.norm_aspect_values = {NormSystem.aspect_domain:[domain_aspect],
                                   NormSystem.aspect_premise:[]}

    def domain_aspect(self):
        return self.norm_aspect_values[NormSystem.aspect_domain][0]

    def add_premise_aspect(self, premise_aspect):
        self.norm_aspect_values[NormSystem.aspect_premise].append(premise_aspect)


class NormAspectValues:
    """
    Encapsules the values of a norm for a specific aspect
    """
    def __init__(self, aspect_name, o_vals, r_vals, s_vals, t_vals, norm_type, h_val, identifier,  starttime):
        """
        Norms are hyperrectangles https://de.wikipedia.org/wiki/Hyperrechteck
        (https://de.wikipedia.org/wiki/Hyperw%C3%BCrfel for visual reference on faces and nodes).

        Initializes the attributes, calculates the vertices and faces of the hyperrectangle.

        :param aspect_name: Name of the corresponding aspect
        :param o_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm_aspect_value
                        for dimensiono
        :param r_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm_aspect_value
                        for dimension r
        :param s_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm_aspect_value
                        for dimension s
        :param t_vals: tuple (x1,x2) with x1 being the first value and x2 the last value on scale included in the norm_aspect_value
                        for dimension t
        :param norm_type: either NormSystem.type_obl, NormSystem.type_perm or NormSystem.type_prohib
        """
        self.aspect_name = aspect_name
        self.start_values = {NormSystem.dim_o: o_vals[0], NormSystem.dim_r: r_vals[0],
                             NormSystem.dim_s: s_vals[0], NormSystem.dim_t: t_vals[0]}
        self.end_values = {NormSystem.dim_o: o_vals[1], NormSystem.dim_r: r_vals[1],
                           NormSystem.dim_s: s_vals[1], NormSystem.dim_t: t_vals[1]}
        self.norm_type = norm_type
        self.identifier = identifier
        self.starttime = starttime
        self.hierarchy = h_val
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
                if o.value != o_v.value:
                    neighbours.append((o_v, r, s, t))
            for r_v in r_vals:
                if r.value != r_v.value:
                    neighbours.append((o, r_v, s, t))
            for s_v in s_vals:
                if s.value != s_v.value:
                    neighbours.append((o, r, s_v, t))
            for t_v in t_vals:
                if t.value != t_v.value:
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
        Calculates the minimal distance between a given case and the norm_aspect_value. Uses Euclidean Distance.
        Could maybe be much faster....

        :param case: The case to check the distance for.
        :return: The calculated minimal distance.
        """
        min_dist = None
        case_vector = get_mathematical_vertex((case.coordinates(self.aspect_name)[NormSystem.dim_o],
                                               case.coordinates(self.aspect_name)[NormSystem.dim_r],
                                               case.coordinates(self.aspect_name)[NormSystem.dim_s],
                                               case.coordinates(self.aspect_name)[NormSystem.dim_t]))
        # calculate minimal distance from any face
        if len(self.faces) == 0:  # happens if a norm_aspect_value has equal start and end point in all dimensions
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
            distance = np.inf
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
                        # check if projection is between the vertices
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
        Checks if a case is subsumed by the norm_aspect_value. Since it's a hyperrectangle, we can simply check whether the
        coordinates of the case are within start and end value of the norm_aspect_value for every dimension.

        :param case: The case to check.
        :return: True if the norm_aspect_value subsumes the case, False otherwise.
        """
        outside = False
        # if the case does not have this aspect, then it does not subsume
        if self.aspect_name not in case.case_aspect_values.keys():
            return False
        for dim in NormSystem.dimensions:
            if not self.start_values[dim].value <= case.coordinates(self.aspect_name)[dim].value <= self.end_values[dim].value:
                outside = True
        return not outside

    def reset_outside_cases(self):
        self.outside_cases = []

    def add_outside_case(self, case):
        """
        Adds a given case to the list of cases not subsumed by the norm_aspect_value. Used for plotting later on.

        :param case: Case to add.
        """
        self.outside_cases.append(case)


class NormAspect:
    """
    Holds the information about a general normAspect - no specific values of a norm
    """
    def __init__(self, norm_type_list, aspect_name, scales):
        self.colors = {}
        self.hatches = {}
        self.contrary_map = {}
        self.type_names = set()
        object_scale, r_scale, subject_scale, time_scale, hierarchy_scale = scales
        self.scales = {NormSystem.dim_o: object_scale, NormSystem.dim_r: r_scale,
                       NormSystem.dim_s: subject_scale, NormSystem.dim_t: time_scale,
                       NormSystem.hierarchy:hierarchy_scale}
        self.aspect_name = aspect_name
        for norm_type in norm_type_list:
            self.type_names.add(norm_type.name)
            self.colors[norm_type.name] = norm_type.color
            self.hatches[norm_type.name] = norm_type.hatch
            if norm_type.contrary_type_name is not None:
                self.contrary_map[norm_type.name] = norm_type.contrary_type_name

    def get_norm_color(self, norm_aspect_value):
        return self.colors[norm_aspect_value.norm_type]

    def get_contrary_norm_color(self, norm_aspect_value):
        return self.colors[self.contrary_map[norm_aspect_value.norm_type]]


class NormType:
    """
    Holds the information about a normtype
    """
    def __init__(self, name, color, hatch, contrary_type_name=None):
        self.name=name
        self.color=color
        self.hatch=hatch
        self.contrary_type_name=contrary_type_name


class DimValue:
    """
    Encapsules a coordinate on one scale. Name is the displayed scale value und value the correspnding mathematical value
    """
    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __eq__(self, other):
        return (self.value == other.value) & (self.name == other.name)

    def __hash__(self):
        return hash(str(self.name)+str(self.value))