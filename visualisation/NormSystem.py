import itertools
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

k_nearest_norms = 3
fig_path = 'figures/'
fig_path_detailes = 'figures_detailed/'


def get_generic_h_scale(num_levels):
    """
    Creates a scale for the hierarchy with given levels. Levels are simply numbered starting with 1.
    Users can also create custom scales, as long as the scale is a list.

    :param num_levels: Number of levels for the hierarchy
    :return: a list of levels, starting with the lowest hierarchy level
    """
    return ['$H_{'+str(i+1)+'}$' for i in range(num_levels)]


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


def add_arrow(ax, start, length, color):
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
    text = '$'+norm.identifier+'$'
    if dim_one:  # add starttime for one dimensional plots
        text = r'$' + norm.identifier + r'_{\mathit{'+norm.starttime+'}}$'
    ax.annotate(text=text,
                xy=(x, y), xytext=(x + NormSystem.annotation_offset, y),
                verticalalignment='center', fontsize=NormSystem.fontsize)


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
    ax.set_yticks([i - 0.5 for i in range(len(scale)+1)])
    ax.yaxis.set_tick_params(length=0)  # length 0 to hide ticks
    ax.set_yticklabels(['']+scale, fontsize=NormSystem.fontsize)

    # draw deviders
    for h in range(len(scale)+1):
        ax.axhline(y=h, color='grey', linestyle='dotted')


def prepare_one_axis(ax, scale, axis_label, x_axis=True):
    """
    Sets up an axis for plotting.

    :param ax: exes of the plot
    :param scale: scale with the names of the axis ticks.
    :param axis_label: Name / label of the axis
    :param x_axis: whether this is the x-axis
    """
    if x_axis:
        ax.set_xlim((-0.5, len(scale)))
    else:
        ax.set_ylim((-0.5, len(scale)))
    label = '$'+axis_label+'$'
    # create scale with ticks in middle
    if x_axis:
        ax.set_xticks([i + 0.5 for i in range(len(scale))])
        ax.set_xticklabels(scale, fontsize=NormSystem.fontsize)
        ax.set_xlabel(label, loc="right", labelpad=0, fontsize=NormSystem.fontsize)
    else:
        ax.set_yticks([i + 0.5 for i in range(len(scale))])
        ax.set_yticklabels(scale, fontsize=NormSystem.fontsize)
        ax.set_ylabel(label, loc='top', labelpad=0, rotation=0,
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
    Adds an analogy rectangle for two dimensional plots.

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


def change_color(norm_type, color_hex):
    if norm_type in [NormSystem.type_obl, NormSystem.type_prohib, NormSystem.type_perm]:
        NormSystem.colors[norm_type] = color_hex
        NormSystem.contrary_map = {NormSystem.type_prohib: NormSystem.colors[NormSystem.type_perm],
                                   NormSystem.type_perm: NormSystem.colors[NormSystem.type_prohib]}


class NormSystem:
    dim_o = 'O'
    dim_r = 'R'
    dim_s = 'S'
    dim_t = 'T'
    type_obl = 'obligation'
    type_perm = 'permission'
    type_prohib = 'prohibition'
    fontsize = 22
    dashed_linestyle = (0, (7, 5))
    linewidth = 2
    annotation_offset = 0.25
    colors = {type_obl: '#9cbad8', type_perm: '#fcde78', type_prohib: '#e66560'}
    # possible hatches: https://stackoverflow.com/questions/14279344/how-can-i-add-textures-to-my-bars-and-wedges
    hatches = {type_obl: '**', type_perm: '/////', type_prohib: '\\\\\\\\\\'}
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
        o_start, o_end = o_vals
        r_start, r_end = r_vals
        s_start, s_end = s_vals
        t_start, t_end = t_vals
        self.norm_list.append(Norm(o_vals=[(o_start, self.get_position(o_start, NormSystem.dim_o)),
                                           (o_end, self.get_position(o_end, NormSystem.dim_o))],
                                   r_vals=[(r_start, self.get_position(r_start, NormSystem.dim_r)),
                                           (r_end, self.get_position(r_end, NormSystem.dim_r))],
                                   s_vals=[(s_start, self.get_position(s_start, NormSystem.dim_s)),
                                           (s_end, self.get_position(s_end, NormSystem.dim_s))],
                                   t_vals=[(t_start, self.get_position(t_start, NormSystem.dim_t)),
                                           (t_end, self.get_position(t_end, NormSystem.dim_t))],
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
            identifier = r'$\mathfrak{C}_{'+str(len(self.case_list)+1)+'}$'
        self.case_list.append(Case(o_val=(o_val, self.get_position(o_val, NormSystem.dim_o)),
                                   r_val=(r_val, self.get_position(r_val, NormSystem.dim_r)),
                                   s_val=(s_val, self.get_position(s_val, NormSystem.dim_s)),
                                   t_val=(t_val, self.get_position(t_val, NormSystem.dim_t)),
                                   identifier=identifier
                                   ))

    def get_relevant_norms(self):
        """
        Is called when a figure in a NormSystem should be plotted. Selects all normas relevant to the cases.
        These are all norms which subsume a case and in case that a case is not subsumed by any norm, the k closest
        norms are returned for that norm.

        :return: a set of all relevant norms.
        """
        for norm in self.norm_list:
            norm.outside_cases=[]
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
        :param dim: x axis from NormSystem.dimensions
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
            offset = 1 / (num_norms+1)
            for i in range(len(norms)):
                current_offset = offset*(i+1)
                norm = norms[i]
                norm_color = NormSystem.colors[norm.norm_type]
                y_position = norm.hierarchy[1] + current_offset
                ax.hlines(y=y_position, xmax=norm.end_values[dim][1] + 1, xmin=norm.start_values[dim][1],
                          color=norm_color,
                          linewidth=NormSystem.linewidth)

                # dashed lines from axes
                dashes_lines_x_points = [norm.start_values[dim][1], norm.end_values[dim][1]+1]

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
                        add_arrow(ax, start=(x_start_line, y_position-(0.3*offset)),
                                  length=(x_min-x_start_line, 0),
                                  color=NormSystem.contrary_map[norm.norm_type])
                        add_arrow(ax, start=(x_end_line, y_position-(0.3*offset)),
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
        prepare_one_axis(scale=scale, axis_label=dim, ax=ax)
        prepare_hierarchy_axis(scale=self.h_scale, ax=ax)

    def draw_two_dim_subplot(self, ax, dim_x, dim_y, detailed):
        """
        Draws one subplot for a two dimensional plot.

        :param ax: axes of the plot
        :param dim_x: x axis from NormSystem.dimensions
        :param dim_y: y axis from NormSystem.dimensions
        :param detailed: whether to draw the detailed version
        """
        # cases
        annotations = {}
        for case in self.case_list:
            annotations = plot_case(ax, x=case.coordinates[dim_x][1], y=case.coordinates[dim_y][1],
                                    annotations=annotations, identifier=case.identifier)
        annotate_cases(ax, annotations)

        # norms
        for norm in self.get_relevant_norms():
            norm_color = NormSystem.colors[norm.norm_type]

            ax.add_patch(Rectangle((norm.start_values[dim_x][1], norm.start_values[dim_y][1]),
                                   norm.end_values[dim_x][1] - norm.start_values[dim_x][1] + 1,
                                   norm.end_values[dim_y][1] - norm.start_values[dim_y][1] + 1,
                                   facecolor='none',
                                   hatch=NormSystem.hatches[norm.norm_type], linewidth=NormSystem.linewidth,
                                   edgecolor=norm_color))
            annotate_norm(ax=ax, norm=norm, x=norm.end_values[dim_x][1] + 1,
                          y=norm.start_values[dim_y][1] + (norm.end_values[dim_y][1] + 1 -
                                                           norm.start_values[dim_y][1]) / 2)

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
                            if x > x_end_rect-1:  # on the right
                                y_offset = -0.5 + min(y_max-y_end_rect, x-(x_end_rect-1))
                            if x < x_start_rect:  # on the left
                                y_offset = -0.5 + min(y_max - y_end_rect, x_start_rect - x)
                            add_arrow(ax, start=(x+0.5, y_end_rect+y_offset),
                                      length=(0, y_max - y_end_rect-y_offset),
                                      color=NormSystem.contrary_map[norm.norm_type])
                    # bottom
                    if y_min < y_start_rect:
                        for x in range(x_min, x_max):
                            y_offset = 0
                            if x > x_end_rect - 1:  # on the right
                                y_offset = -0.5 + min(y_start_rect - y_min, x - (x_end_rect - 1))
                            if x < x_start_rect:  # on the left
                                y_offset = -0.5 + min(y_start_rect - y_min, x_start_rect - x)
                            add_arrow(ax, start=(x+0.5, y_start_rect-y_offset),
                                      length=(0, y_min - y_start_rect+y_offset),
                                      color=NormSystem.contrary_map[norm.norm_type])
                    # right
                    if x_max > x_end_rect:
                        for y in range(y_min, y_max):
                            x_offset = 0
                            if y > y_end_rect - 1:  # at the top
                                x_offset = -0.5 + min(x_max - x_end_rect, y - (y_end_rect - 1))
                            if y < y_start_rect:  # at the bottom
                                x_offset = - 0.5 + min(x_max - x_end_rect, y_start_rect - y)
                            add_arrow(ax, start=(x_end_rect+x_offset, y + 0.5),
                                      length=(x_max - x_end_rect-x_offset, 0),
                                      color=NormSystem.contrary_map[norm.norm_type])
                    # left
                    if x_min < x_start_rect:
                        for y in range(y_min, y_max):
                            x_offset = 0
                            if y > y_end_rect - 1:  # at the top
                                x_offset = -0.5 + min(x_start_rect - x_min, y - (y_end_rect - 1))
                            if y < y_start_rect:  # at the bottom
                                x_offset = - 0.5 + min(x_start_rect - x_min, y_start_rect - y)
                            add_arrow(ax, start=(x_start_rect-x_offset, y + 0.5),
                                      length=(x_min - x_start_rect+x_offset, 0),
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

        # arrows on axes
        ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False)
        ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_linewidth(NormSystem.linewidth)
        prepare_one_axis(scale=self.scales[dim_x], axis_label=dim_x, ax=ax)
        prepare_one_axis(scale=self.scales[dim_y], axis_label=dim_y, ax=ax, x_axis=False)

    def draw_dims_one(self, saveidentifier, detailed):
        """
        Coordinates drawing figures for one dimenions each.

        :param saveidentifier: an identifier to include in the name of the saved file.
        :param detailed:  Whether to draw the detailled version of the figures.
        """
        if detailed:
            savename = fig_path_detailes + saveidentifier + '_dims_one.png'
        else:
            savename = fig_path + saveidentifier + '_dims_one.png'
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        fig = plt.figure(figsize=(10, 20))  # (15,30) also good
        ax1 = fig.add_subplot(411)  # Plot with: 4 rows, 1 column
        self.draw_one_dim_subplot(ax1, NormSystem.dim_o, detailed)
        ax2 = fig.add_subplot(412)
        self.draw_one_dim_subplot(ax2, NormSystem.dim_r, detailed)
        ax3 = fig.add_subplot(413)
        self.draw_one_dim_subplot(ax3, NormSystem.dim_s, detailed)
        ax4 = fig.add_subplot(414)
        self.draw_one_dim_subplot(ax4, NormSystem.dim_t, detailed)

        fig.tight_layout()
        plt.savefig(savename)
        plt.close()

    def draw_dims_two(self, dims_one, dims_two, saveidentifier, detailed):
        """
        Coordinates drawing figures for two dimenions each.

        :param dims_one: (dim_x, dim_y) for first plot (from NormSystem.dimensions)
        :param dims_two: (dim_x, dim_y) for second plot (from NormSystem.dimensions)
        :param saveidentifier: an identifier to include in the name of the saved file.
        :param detailed: Whether to draw the detailled version of the figures.
        """
        if detailed:
            savename = fig_path_detailes+saveidentifier+'_dims_two.png'
        else:
            savename = fig_path + saveidentifier + '_dims_two.png'
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)  # Plot with: 1 row, 2 column, first subplot.
        self.draw_two_dim_subplot(ax1, dims_one[0], dims_one[1], detailed)
        ax2 = fig.add_subplot(122)  # Plot with: 1 row, 2 column, second subplot.
        self.draw_two_dim_subplot(ax2, dims_two[0], dims_two[1], detailed)

        fig.tight_layout()
        plt.savefig(savename)
        plt.close()

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
    return np.array([vertice_with_names[0][1], vertice_with_names[1][1],
                     vertice_with_names[2][1], vertice_with_names[3][1]])


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
        # calculate minimal distance from any face
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

            case_vector = get_mathematical_vertex((case.coordinates[NormSystem.dim_o],
                                                   case.coordinates[NormSystem.dim_r],
                                                   case.coordinates[NormSystem.dim_s],
                                                   case.coordinates[NormSystem.dim_t]))
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
                        # orthogonal projection on line
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
