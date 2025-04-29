import ctypes

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QRegularExpression, Qt, QSize
from PyQt5.QtGui import QRegularExpressionValidator, QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QGridLayout, QVBoxLayout, QLineEdit, \
    QLabel, QPushButton, QComboBox, QCheckBox, QScrollArea, QButtonGroup
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import sys

from visualisation.NormSystem import NormSystem, get_generic_dim_scale, get_generic_h_scale, reset_colors, change_color

dim_x_str = 'X'
dim_y_str = 'Y'
none_selected_str = 'None selected'
display_names = {none_selected_str: none_selected_str,
                 NormSystem.dim_o: 'Dimension O: Objects', NormSystem.dim_r: 'Dimension R: Space',
                 NormSystem.dim_s: 'Dimension S: Subjects', NormSystem.dim_t: 'Dimension T: Time',
                 NormSystem.hierarchy: 'Hierarchy'}

title_font = QFont('Arial', 12)
normal_font = QFont('Arial', 8)
one_row_normal_label_height = 20
one_letter_normal_label_width = 15


def latex_to_qpixmap(text):
    """
    Creates an image using matpoltlib for the latex and covnerts it to a qpixmap.

    :param text: Latex text to convert
    :return: created pixmap
    """
    fig = figure.Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()
    # ---- plot the text expression ----
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, text, ha='left', va='bottom', fontsize=15)
    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)
    text_bbox = t.get_window_extent(renderer)
    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height
    fig.set_size_inches(tight_fwidth, tight_fheight)
    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                  QtGui.QImage.Format_ARGB32))
    qpixmap = QtGui.QPixmap(qimage)
    return qpixmap



def get_button(text):
    """
    Creates a button and does the styling etc
    :param text: Text to display on the button
    :return: the created button
    """
    button = QPushButton(text)
    button.setFont(normal_font)
    return button


def setup_input_field(line_edit, validator_regex, placeholder_text):
    """
    Sets up a given QLineEdit. Does the styling etc.

    :param line_edit: widget to set up
    :param validator_regex: Regex for input validation
    :param placeholder_text: Text to be displayed as placeholder
    """
    input_validator = QRegularExpressionValidator(
        QRegularExpression(validator_regex), line_edit)
    line_edit.setValidator(input_validator)
    line_edit.setPlaceholderText(placeholder_text)
    line_edit.setFont(normal_font)


def get_normal_label(text):
    """
    Creates a label with normal text size.

    :param text: Text for the label.
    :return: The created label
    """
    title_label = QLabel(text)
    title_label.setFixedHeight(one_row_normal_label_height)
    title_label.setFont(normal_font)
    return title_label


def get_one_letter_normal_label(text):
    """
    Creates a label for normal text with one letter and small fixed width

    :param text: Text to be displayed
    :return: The created label
    """
    label = get_normal_label(text)
    label.setFixedWidth(one_letter_normal_label_width)
    return label


def get_title_label(text):
    """
    Creates a label for Title text.

    :param text: The Text to display
    :return: The created label
    """
    title_label = QLabel(text)
    title_label.setFixedHeight(35)
    title_label.setFont(title_font)
    return title_label


def reset_combo_box(box, pix_map, values):
    """
    Resets a combo box (removes all values) and adds new values. Intended for displaying the text as latex.

    :param box: The combobox to work on.
    :param pix_map: The pixmap to find the values in.
    :param values: The values to set
    """
    box.clear()
    box.view().setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    box.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    height = 0
    width = 0
    for i in range(len(values)):
        val = values[i]
        pixmap = pix_map[val]
        height = max(height, pixmap.height())
        width = max(width, pixmap.width())
        box.addItem(QIcon(pixmap), '')
        box.setItemData(i, val, Qt.ItemDataRole.UserRole)
    size = QSize(width, height)
    box.setIconSize(size)


def set_pixmap(pixmap, values):
    """
    Fills the pixmap from the given values.

    :param pixmap: pixmap to vill (dict)
    :param values: String values (list)
    """
    for v in values:
        if v not in pixmap.keys():
            pixmap[v] = latex_to_qpixmap(v)


def clear_layout(layout):
    """
    Removes all children of a layout

    :param layout: layout to remove children from
    """
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clear_layout(item.layout())


def insert_linebreaks_in_tick_label(text):
    words = text.split()
    rough_max_length = 15
    current_length = 0
    res = []
    for i in range(len(words)):
        current_length += len(words[i])
        if current_length > rough_max_length:
            res.append(words[i]+'\n')
            current_length = 0
        else:
            res.append(words[i])
    return ' '.join(res)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ViNo - Visualisation of Norms")
        self.setWindowIcon(QtGui.QIcon('logo_vino.png'))
        self.setMinimumSize(1400, 900)
        self.norm_system = None
        self.detailed = True
        self.x_axis = NormSystem.dim_o
        self.y_axis = none_selected_str
        self.show_all_one_dim = False
        self.show_both_two_dims = False
        self.o_pixmap = {}
        self.r_pixmap = {}
        self.s_pixmap = {}
        self.t_pixmap = {}
        self.h_pixmap = {}

        self.dim_raw_texts = {NormSystem.dim_o: '', NormSystem.dim_r: '', NormSystem.dim_s: '',
                              NormSystem.dim_t: '', NormSystem.hierarchy: ''}
        self.norm_system_error = get_normal_label('')
        self.norm_system_error.setFixedHeight(0)
        self.norm_system_o_input = QLineEdit()
        self.norm_system_r_input = QLineEdit()
        self.norm_system_s_input = QLineEdit()
        self.norm_system_t_input = QLineEdit()
        self.norm_system_h_input = QLineEdit()
        self.combo_y_dim = QComboBox()
        self.checkbox_all = QCheckBox('Display other combination')
        self.colors_norms_type = QComboBox()
        self.colors_hex_input = QLineEdit()
        self.cases_o_combo = QComboBox()
        self.cases_r_combo = QComboBox()
        self.cases_s_combo = QComboBox()
        self.cases_t_combo = QComboBox()
        self.case_name_edit = QLineEdit()
        self.norms_o_start_combo = QComboBox()
        self.norms_o_end_combo = QComboBox()
        self.norms_r_start_combo = QComboBox()
        self.norms_r_end_combo = QComboBox()
        self.norms_s_start_combo = QComboBox()
        self.norms_s_end_combo = QComboBox()
        self.norms_t_start_combo = QComboBox()
        self.norms_t_end_combo = QComboBox()
        self.norms_hierarchy = QComboBox()
        self.norms_type = QComboBox()
        self.norms_starttime = QLineEdit()
        self.norms_identifier = QLineEdit()
        self.norms_input_error = QLabel()
        self.norm_list = QVBoxLayout()
        self.case_list = QVBoxLayout()
        self.norm_scroll = QScrollArea()
        self.case_scroll = QScrollArea()
        self.case_button_group = QButtonGroup()
        self.norm_button_group = QButtonGroup()

        layout_main = QHBoxLayout()
        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()
        layout_main.addLayout(layout_left, 2)
        layout_main.addLayout(layout_right, 3)

        # Norm System, upper left
        layout_norm_sytem = QVBoxLayout()
        self.setup_norm_system_layout(layout_norm_sytem)
        layout_left.addLayout(layout_norm_sytem)

        # Norms & Cases, lower left
        layout_norms = QVBoxLayout()
        self.setup_norms_layout(layout_norms)
        layout_left.addLayout(layout_norms)

        layout_cases = QVBoxLayout()
        self.setup_cases_layout(layout_cases)
        layout_left.addLayout(layout_cases)

        # Dimensions, upper right
        layout_dimension_settings = QVBoxLayout()
        self.setup_dimension_settings_layout(layout_dimension_settings)
        layout_right.addLayout(layout_dimension_settings)

        # Image, lower right
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        layout_right.addWidget(toolbar)
        layout_right.addWidget(self.canvas)

        widget = QWidget()
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)

    def setup_norms_layout(self, layout):
        """
        Creates the layout for creating, displaying and deleting norms.

        :param layout: Layout to use
        """
        layout.addWidget(get_title_label('Norms'))

        line_one = QHBoxLayout()
        o_input = QHBoxLayout()
        o_input.addWidget(get_one_letter_normal_label(NormSystem.dim_o))
        o_input.addWidget(self.norms_o_start_combo)
        o_input.addWidget(self.norms_o_end_combo)
        line_one.addLayout(o_input, 1)

        r_input = QHBoxLayout()
        r_input.addWidget(get_one_letter_normal_label(NormSystem.dim_r))
        r_input.addWidget(self.norms_r_start_combo)
        r_input.addWidget(self.norms_r_end_combo)
        line_one.addLayout(r_input, 1)

        line_two = QHBoxLayout()
        s_input = QHBoxLayout()
        s_input.addWidget(get_one_letter_normal_label(NormSystem.dim_s))
        s_input.addWidget(self.norms_s_start_combo)
        s_input.addWidget(self.norms_s_end_combo)
        line_two.addLayout(s_input, 1)

        t_input = QHBoxLayout()
        t_input.addWidget(get_one_letter_normal_label(NormSystem.dim_t))
        t_input.addWidget(self.norms_t_start_combo)
        t_input.addWidget(self.norms_t_end_combo)
        line_two.addLayout(t_input, 1)

        line_three = QHBoxLayout()
        h_input = QHBoxLayout()
        h_input.addWidget(get_one_letter_normal_label(NormSystem.hierarchy))
        h_input.addWidget(self.norms_hierarchy, 1)
        self.norms_type.addItems([NormSystem.type_obl, NormSystem.type_perm, NormSystem.type_prohib])
        h_input.addWidget(self.norms_type, 1)
        line_three.addLayout(h_input, 1)

        id_input = QHBoxLayout()
        self.norms_starttime.setPlaceholderText('Temporal indicator')
        id_input.addWidget(self.norms_starttime)
        self.norms_identifier.setPlaceholderText('Identifier')
        id_input.addWidget(self.norms_identifier)
        line_three.addLayout(id_input, 1)

        layout.addLayout(line_one)
        layout.addLayout(line_two)
        layout.addLayout(line_three)

        add_norm_button = get_button('Add Norm')
        add_norm_button.clicked.connect(self.add_norm)
        layout.addWidget(add_norm_button)
        layout.addWidget(self.norms_input_error)
        self.norms_input_error.setFixedHeight(0)
        norm_list_widget = QWidget()
        norm_list_widget.setLayout(self.norm_list)

        self.norm_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.norm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.norm_scroll.setWidgetResizable(True)
        self.norm_list.addStretch()
        self.norm_scroll.setWidget(norm_list_widget)

        layout.addWidget(self.norm_scroll)

    def setup_cases_layout(self, layout):
        """
        Creates the layout for creating, listing and deleting cases.

        :param layout: The layout to use
        """
        layout.addWidget(get_title_label('Cases'))
        input_layout = QGridLayout()

        input_layout.addWidget(self.cases_o_combo, 0, 0)
        input_layout.addWidget(self.cases_r_combo, 0, 1)
        input_layout.addWidget(self.cases_s_combo, 0, 2)
        input_layout.addWidget(self.cases_t_combo, 0, 3)

        layout.addLayout(input_layout)

        self.case_name_edit.setPlaceholderText('Identifier')
        layout.addWidget(self.case_name_edit)

        button_add_case = get_button('Add Case')
        button_add_case.clicked.connect(self.add_case)
        layout.addWidget(button_add_case)

        case_list_widget = QWidget()
        case_list_widget.setLayout(self.case_list)

        self.case_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.case_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.case_scroll.setWidgetResizable(True)
        self.case_list.addStretch()
        self.case_scroll.setWidget(case_list_widget)

        layout.addWidget(self.case_scroll)

    def add_norm(self):
        """
        Adds a norm if the entered values are correct. Displays the errors otherwise.
        """
        if self.norms_hierarchy.count() > 0:
            errortext = []
            norm_id = self.norms_identifier.text()
            if norm_id.strip() == '':
                errortext.append('Identifier must be given')
            starttime = self.norms_starttime.text()
            if starttime.strip() == '':
                errortext.append('Temporal indicator must be given')
            if len(errortext) > 0:
                self.norms_input_error.setText('\n'.join(errortext))
                self.norms_input_error.setFixedHeight(len(errortext) * one_row_normal_label_height)
            else:
                self.norms_input_error.setText('')
                self.norms_input_error.setFixedHeight(0)
                self.norm_system.add_norm(o_vals=(self.norms_o_start_combo.currentData(Qt.ItemDataRole.UserRole),
                                                  self.norms_o_end_combo.currentData(Qt.ItemDataRole.UserRole)),
                                          r_vals=(self.norms_r_start_combo.currentData(Qt.ItemDataRole.UserRole),
                                                  self.norms_r_end_combo.currentData(Qt.ItemDataRole.UserRole)),
                                          s_vals=(self.norms_s_start_combo.currentData(Qt.ItemDataRole.UserRole),
                                                  self.norms_s_end_combo.currentData(Qt.ItemDataRole.UserRole)),
                                          t_vals=(self.norms_t_start_combo.currentData(Qt.ItemDataRole.UserRole),
                                                  self.norms_t_end_combo.currentData(Qt.ItemDataRole.UserRole)),
                                          hierarchy=self.norms_hierarchy.currentData(Qt.ItemDataRole.UserRole),
                                          starttime=starttime,
                                          norm_type=self.norms_type.currentText(),
                                          identifier=norm_id)
                self.update_norm_list()
                self.plot_current_setup()

    def update_norm_list(self):
        """
        Updates the list showing the norms. Removes the old entries and inserts current norms.
        """
        clear_layout(self.norm_list)
        self.norm_list.addStretch()
        self.norm_button_group = QButtonGroup()
        for i in range(len(self.norm_system.norm_list)):
            norm = self.norm_system.norm_list[i]
            widget = QWidget()
            h_layout = QHBoxLayout()
            delete_button = get_button('x')
            delete_button.setMaximumSize(25, 25)
            self.norm_button_group.addButton(delete_button, i)
            h_layout.addWidget(delete_button)

            latex_expression = '$ \\mathbf{' + norm.identifier + '} ('+norm.norm_type+'): ($' + \
                               norm.start_values[NormSystem.dim_o][0].replace('\n','') + '$,$' + \
                               norm.end_values[NormSystem.dim_o][0].replace('\n','') + \
                               '$), ($' + norm.start_values[NormSystem.dim_r][0].replace('\n','') + '$,$' + \
                               norm.end_values[NormSystem.dim_r][0].replace('\n','') + \
                               '$), ($' + norm.start_values[NormSystem.dim_s][0].replace('\n','') + '$,$' + \
                               norm.end_values[NormSystem.dim_s][0].replace('\n','') + \
                               '$), ($' + norm.start_values[NormSystem.dim_t][0].replace('\n','') + '$,$' + \
                               norm.end_values[NormSystem.dim_t][0].replace('\n','') + '$), $' +\
                               norm.hierarchy[0].replace('\n','') +\
                               '$, '+norm.starttime+'$'
            pix_map = latex_to_qpixmap(latex_expression)
            norm_label = QLabel()
            norm_label.setPixmap(pix_map)
            h_layout.addWidget(norm_label)

            h_layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(h_layout)
            self.norm_list.insertWidget(self.norm_list.count() - 1, widget)
        self.norm_button_group.buttonClicked.connect(self.delete_norm)

    def update_case_list(self):
        """
        Updates the list showing the cases. Removes the old entries and inserts the current cases.
        """
        clear_layout(self.case_list)
        self.case_list.addStretch()
        self.case_button_group = QButtonGroup()
        for i in range(len(self.norm_system.case_list)):
            case = self.norm_system.case_list[i]
            widget = QWidget()
            h_layout = QHBoxLayout()

            delete_button = get_button('x')
            delete_button.setMaximumSize(25, 25)
            self.case_button_group.addButton(delete_button, i)
            h_layout.addWidget(delete_button)

            latex_expression = '$ \\mathbf{' + case.identifier.replace('$', '') + '}: $' + \
                               case.coordinates[NormSystem.dim_o][0].replace('\n','') + \
                               '$, $' + case.coordinates[NormSystem.dim_r][0].replace('\n','') + \
                               '$, $' + case.coordinates[NormSystem.dim_s][0].replace('\n','') + \
                               '$, $' + case.coordinates[NormSystem.dim_t][0].replace('\n','')
            pix_map = latex_to_qpixmap(latex_expression)
            case_label = QLabel()
            case_label.setPixmap(pix_map)
            h_layout.addWidget(case_label)

            h_layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(h_layout)
            self.case_list.insertWidget(self.case_list.count() - 1, widget)
        self.case_button_group.buttonClicked.connect(self.delete_case)

    def delete_case(self, button):
        """
        Method to connect to the button of a case for deletion. Deletes the case of the button.

        :param button: The clicked button.
        """
        case = None
        for i in range(len(self.norm_system.case_list)):
            if self.case_button_group.button(i) == button:
                case = self.norm_system.case_list[i]
        if case is not None:
            self.norm_system.delete_case(case)
            self.plot_current_setup()
            self.update_case_list()

    def delete_norm(self, button):
        """
        Method to connect to the button of a norm for deletion. Deletes the norm of the button.

        :param button: The clicked button.
        """
        norm = None
        for i in range(len(self.norm_system.norm_list)):
            if self.norm_button_group.button(i) == button:
                norm = self.norm_system.norm_list[i]
        if norm is not None:
            self.norm_system.delete_norm(norm)
            self.plot_current_setup()
            self.update_norm_list()

    def add_case(self):
        """
        Adds a case to the norm system. If no identifier is given, a standard name is used.
        """
        if self.cases_s_combo.count() > 0:
            case_name = self.case_name_edit.text()
            if case_name.strip() == '':
                case_name = None
            self.norm_system.add_case(o_val=self.cases_o_combo.currentData(Qt.ItemDataRole.UserRole),
                                      r_val=self.cases_r_combo.currentData(Qt.ItemDataRole.UserRole),
                                      s_val=self.cases_s_combo.currentData(Qt.ItemDataRole.UserRole),
                                      t_val=self.cases_t_combo.currentData(Qt.ItemDataRole.UserRole),
                                      identifier=case_name)
            self.update_case_list()
            self.plot_current_setup()

    def setup_dimension_settings_layout(self, layout):
        """
        Creates the layout for the settings. Sets colors and axes for dimensions.

        :param layout: The layout to use
        """
        layout.addWidget(get_title_label('Settings'))

        dim_settings_layout = QHBoxLayout()
        # x axis
        dim_x_layout = QVBoxLayout()
        dim_x_layout.addWidget(get_normal_label('Select dimension for x-axis.'))

        x_select = QHBoxLayout()
        x_select.addWidget(get_one_letter_normal_label(dim_x_str))

        combo_x_dim = QComboBox()
        combo_x_dim.addItems([display_names[NormSystem.dim_o], display_names[NormSystem.dim_r],
                              display_names[NormSystem.dim_s], display_names[NormSystem.dim_t]])
        combo_x_dim.currentTextChanged.connect(lambda state: self.update_axis(state, dim_x_str))
        x_select.addWidget(combo_x_dim)
        dim_x_layout.addLayout(x_select)

        checkbox_all = QCheckBox('Display all dimensions')
        checkbox_all.clicked.connect(self.update_show_all_one_dim)
        dim_x_layout.addWidget(checkbox_all)

        dim_settings_layout.addLayout(dim_x_layout, 1)

        # y- axis
        dim_y_layout = QVBoxLayout()
        dim_y_layout.addWidget(get_normal_label('Select dimension for y-axis.'))

        y_select = QHBoxLayout()
        y_select.addWidget(get_one_letter_normal_label(dim_y_str))

        self.combo_y_dim.addItems([v for v in display_names.values()
                                   if v != display_names[NormSystem.hierarchy] and v != combo_x_dim.currentText()])
        self.combo_y_dim.currentTextChanged.connect(lambda state: self.update_axis(state, dim_y_str))
        y_select.addWidget(self.combo_y_dim)
        dim_y_layout.addLayout(y_select)

        self.checkbox_all.clicked.connect(self.update_show_both_two_dims)
        dim_y_layout.addWidget(self.checkbox_all)
        dim_settings_layout.addLayout(dim_y_layout, 1)
        layout.addLayout(dim_settings_layout)

        color_settings_layout = QVBoxLayout()
        heading_layout = QHBoxLayout()
        heading_layout.addWidget(get_normal_label('Change color of norm type'), 1)
        reset_button = get_button('Reset Colors')
        reset_button.clicked.connect(self.reset_all_colors)
        heading_layout.addWidget(reset_button, 1)
        color_settings_layout.addLayout(heading_layout)

        selection_layout = QHBoxLayout()
        setup_input_field(line_edit=self.colors_hex_input, validator_regex='^#(?:[0-9a-fA-F]{6})$',
                          placeholder_text='Hex color code')
        selection_layout.addWidget(self.colors_hex_input, 1)
        self.colors_norms_type.addItems([NormSystem.type_obl, NormSystem.type_perm, NormSystem.type_prohib])
        selection_layout.addWidget(self.colors_norms_type, 1)

        change_button = get_button('Set Color')
        change_button.clicked.connect(self.change_one_color)
        selection_layout.addWidget(change_button, 2)
        color_settings_layout.addLayout(selection_layout)
        layout.addLayout(color_settings_layout)

    def change_one_color(self):
        """
        Sets a new color to one of the norm types
        """
        change_color(self.colors_norms_type.currentText(), self.colors_hex_input.text())
        self.plot_current_setup()

    def reset_all_colors(self):
        """
        Resets the color mapping of the norm types to the original mapping
        """
        reset_colors()
        self.plot_current_setup()

    def update_show_both_two_dims(self, value):
        """
        Reacts to (de-)selecting the option to display all four dimensions for the two-dimensional case (2 figures)
        :param value: (bool) The set value
        """
        self.show_both_two_dims = value
        self.plot_current_setup()

    def update_show_all_one_dim(self, value):
        """
        Reacts to (de-)selecting the option to show all dimensions while displaying one-dimensional figures
        (4 figures in total).

        :param value: (bool) the set value
        """
        self.show_all_one_dim = value
        self.combo_y_dim.setDisabled(value)
        self.checkbox_all.setDisabled(value)
        self.plot_current_setup()

    def update_axis(self, value, dimension):
        """
        Reacts to setting an axis. Sets the internal value and, in case the x axis was set, that dimension is removed
        for the choice of the y axis.

        :param value: The new value
        :param dimension: the dimension
        """
        if value != '':  # happens if combobox is cleared
            key = list(filter(lambda x: display_names[x] == value, display_names))[0]
            if dimension == dim_x_str:
                self.x_axis = key
                self.combo_y_dim.clear()
                self.combo_y_dim.addItems([v for v in display_names.values() if v != value
                                           and v != display_names[NormSystem.hierarchy]])
                self.y_axis = self.combo_y_dim.currentText()

            if dimension == dim_y_str:
                self.y_axis = key
            self.plot_current_setup()

    def setup_norm_system_layout(self, layout):
        """
        Creates the layout for inputting a norm system.

        :param layout: root layout
        """
        input_layout = QGridLayout()
        layout.addWidget(get_title_label('Norm System'))
        layout.addLayout(input_layout)

        input_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[NormSystem.dim_o],
                                                                   dim_id=NormSystem.dim_o,
                                                                   line_edit=self.norm_system_o_input), 0, 0)
        input_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[NormSystem.dim_r],
                                                                   dim_id=NormSystem.dim_r,
                                                                   line_edit=self.norm_system_r_input), 0, 1)
        input_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[NormSystem.dim_s],
                                                                   dim_id=NormSystem.dim_s,
                                                                   line_edit=self.norm_system_s_input), 1, 0)
        input_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[NormSystem.dim_t],
                                                                   dim_id=NormSystem.dim_t,
                                                                   line_edit=self.norm_system_t_input), 1, 1)
        input_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[NormSystem.hierarchy],
                                                                   dim_id=NormSystem.hierarchy,
                                                                   line_edit=self.norm_system_h_input), 2, 0)
        button_dummy_data = get_button('Insert Dummy Data')
        button_dummy_data.clicked.connect(self.insert_dummy_norm_sys)
        input_layout.addWidget(button_dummy_data)

        button_set_norm_system = get_button('Set Values')
        button_set_norm_system.clicked.connect(self.create_norm_system)
        layout.addWidget(button_set_norm_system)
        layout.addWidget(self.norm_system_error)

    def insert_dummy_norm_sys(self):
        """
        Inserts dummy data in the line edits for creating a norm system.
        """
        self.norm_system_o_input.setText(', '.join(get_generic_dim_scale(dim=NormSystem.dim_o, num_entries=9))
                                         .replace('$', '').replace('{', '').replace('}', ''))
        self.norm_system_r_input.setText(', '.join(get_generic_dim_scale(dim=NormSystem.dim_r, num_entries=4))
                                         .replace('$', '').replace('{', '').replace('}', ''))
        self.norm_system_s_input.setText(', '.join(get_generic_dim_scale(dim=NormSystem.dim_s, num_entries=7))
                                         .replace('$', '').replace('{', '').replace('}', ''))
        self.norm_system_t_input.setText(', '.join(get_generic_dim_scale(dim=NormSystem.dim_t, num_entries=5))
                                         .replace('$', '').replace('{', '').replace('}', ''))
        self.norm_system_h_input.setText(', '.join(get_generic_h_scale(num_levels=3))
                                         .replace('$', '').replace('{', '').replace('}', ''))

    def set_dim_raw_text(self, dim, text):
        """
        Sets the raw text for the given dimension

        :param dim: Given dimension
        :param text: raw text data (to be split into list)
        """
        self.dim_raw_texts[dim] = text

    def set_up_norm_system_input_field(self, placeholder_text, dim_id, line_edit):
        """
        Sets up one input layout for a norm system dimension, which consists of a QLabel for the dimension indicator
        and a QLineEdit for inputting values

        :param placeholder_text: Placeholder Text for the input fields
        :param dim_id: Dimension indicator
        :param line_edit: the line_edit to use
        :return: the finished layout
        """
        layout = QHBoxLayout()
        layout.addWidget(get_one_letter_normal_label(dim_id))
        regex_dim_input = "[A-Za-z1-90<=>${_-} ]+(, [A-Za-z1-90<=>${_-} ]+)*"
        setup_input_field(line_edit=line_edit, placeholder_text=placeholder_text, validator_regex=regex_dim_input)
        line_edit.textChanged.connect(lambda state: self.set_dim_raw_text(dim=dim_id, text=state))
        layout.addWidget(line_edit)
        return layout

    def derive_scale(self, dimension):
        """
        Derives the scales from the fiven raw text by splitting it.

        :param dimension: dimension to derive
        :return: scale, error_messages with scale the list of split values anr error_messages contianing possible errors
        """
        scale = self.dim_raw_texts[dimension].split(', ')
        scale = [' '.join(['$' + word + '$' for word in val.split()]) for val in scale if val != '']
        scale = [insert_linebreaks_in_tick_label(entry) for entry in scale]
        num_entries = len(scale)
        error_messages = []
        if num_entries == 0:
            error_messages.append('Dimension ' + dimension + ': No entries given.')
        if len(set(scale)) != num_entries:
            error_messages.append('Dimension ' + dimension + ': Duplicate entries are given.')
        return scale, error_messages

    def create_norm_system(self):
        """
        Creates a NormSystem, if the given values are correct. Otherwise displays an error.
        """
        all_errors = []
        o_scale, errors = self.derive_scale(NormSystem.dim_o)
        all_errors.extend(errors)
        r_scale, errors = self.derive_scale(NormSystem.dim_r)
        all_errors.extend(errors)
        s_scale, errors = self.derive_scale(NormSystem.dim_s)
        all_errors.extend(errors)
        t_scale, errors = self.derive_scale(NormSystem.dim_t)
        all_errors.extend(errors)
        h_scale, errors = self.derive_scale(NormSystem.hierarchy)
        all_errors.extend(errors)

        if len(all_errors) > 0:
            self.norm_system_error.setText('\n'.join(all_errors))
            self.norm_system_error.setFixedHeight(len(all_errors) * one_row_normal_label_height)
        else:
            self.norm_system_error.setText('')
            self.norm_system_error.setFixedHeight(0)
            self.norm_system = NormSystem(object_scale=o_scale, r_scale=r_scale, subject_scale=s_scale,
                                          time_scale=t_scale, hierarchy_scale=h_scale)
            set_pixmap(self.o_pixmap, o_scale)
            set_pixmap(self.r_pixmap, r_scale)
            set_pixmap(self.s_pixmap, s_scale)
            set_pixmap(self.t_pixmap, t_scale)
            set_pixmap(self.h_pixmap, h_scale)

            reset_combo_box(self.cases_o_combo, self.o_pixmap, o_scale)
            reset_combo_box(self.norms_o_end_combo, self.o_pixmap, o_scale)
            reset_combo_box(self.norms_o_start_combo, self.o_pixmap, o_scale)
            reset_combo_box(self.cases_r_combo, self.r_pixmap, r_scale)
            reset_combo_box(self.norms_r_end_combo, self.r_pixmap, r_scale)
            reset_combo_box(self.norms_r_start_combo, self.r_pixmap, r_scale)
            reset_combo_box(self.cases_s_combo, self.s_pixmap, s_scale)
            reset_combo_box(self.norms_s_end_combo, self.s_pixmap, s_scale)
            reset_combo_box(self.norms_s_start_combo, self.s_pixmap, s_scale)
            reset_combo_box(self.cases_t_combo, self.t_pixmap, t_scale)
            reset_combo_box(self.norms_t_end_combo, self.t_pixmap, t_scale)
            reset_combo_box(self.norms_t_start_combo, self.t_pixmap, t_scale)
            reset_combo_box(self.norms_hierarchy, self.h_pixmap, h_scale)

            self.plot_current_setup()
            self.update_norm_list()
            self.update_case_list()

    def plot_current_setup(self):
        """
        Plots the data entered to the NormSystem.
        """
        if self.norm_system is not None:
            self.figure.clear()
            if self.show_all_one_dim:
                self.norm_system.draw_dims_one_all(figure=self.figure, detailed=self.detailed)
            elif self.show_both_two_dims and self.y_axis != none_selected_str:
                dims_one = (self.x_axis, self.y_axis)
                other_dims = [key for key in display_names
                              if key not in [self.y_axis, self.x_axis, NormSystem.hierarchy, none_selected_str]]
                dims_two = (other_dims[0], other_dims[1])
                self.norm_system.draw_dims_two_all(figure=self.figure, detailed=self.detailed,
                                                   dims_one=dims_one, dims_two=dims_two)
            elif self.y_axis != none_selected_str:
                self.norm_system.draw_dims_two(fig=self.figure, detailed=self.detailed,
                                               dim_x=self.x_axis, dim_y=self.y_axis)
            else:
                self.norm_system.draw_dims_one(fig=self.figure, detailed=self.detailed, x_dim=self.x_axis)
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myappid = 'vino.gui.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    window = MainWindow()
    window.show()

    app.exec()
