import ctypes
import os
import random
import weakref

from PyQt6 import QtGui, QtCore
from PyQt6.QtCore import QRegularExpression, Qt, QSize, QUrl, QThread
from PyQt6.QtGui import QRegularExpressionValidator, QFont, QIcon
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, \
    QLabel, QPushButton, QComboBox, QCheckBox, QScrollArea,  QTabWidget, QToolBox, QFrame

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import sys

from matplotlib.figure import Figure

from visualisation.NormSystem import NormSystem, get_generic_dim_scale, CaseAspectValues, DimValue, \
    NormAspectValues, display_types, display_interactive, Norm

dim_x_str = 'X'
dim_y_str = 'Y'
dim_z_str = 'Z'
none_selected_str = 'None selected'
display_names = {none_selected_str: none_selected_str,
                 NormSystem.dim_o: 'Dimension O: Objects', NormSystem.dim_r: 'Dimension R: Space',
                 NormSystem.dim_s: 'Dimension S: Subjects', NormSystem.dim_t: 'Dimension T: Time',
                 NormSystem.hierarchy: 'Hierarchy'}

title_font = QFont('Arial', 12)
normal_font = QFont('Arial', 8)
one_row_normal_label_height = 20
one_letter_normal_label_width = 15
basedir = os.path.dirname(__file__)


def latex_to_qpixmap(text):
    """
    Creates an image using matpoltlib for the latex and covnerts it to a qpixmap.

    :param text: Latex text to convert
    :return: created pixmap
    """
    fig = Figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()
    # ---- plot the text expression ----
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, text, ha='left', va='bottom', fontsize=30)
    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)
    text_bbox = t.get_window_extent(renderer)
    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height
    fig.set_size_inches(tight_fwidth, tight_fheight)
    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(QtGui.QImage(buf, size[0], size[1],
                                                  QtGui.QImage.Format.Format_ARGB32))
    qpixmap = QtGui.QPixmap(qimage)
    qpixmap = qpixmap.scaledToHeight(int(one_row_normal_label_height*0.75), Qt.TransformationMode.SmoothTransformation)
    return qpixmap


def get_horizontal_separator():
    """
    Creates a simple horizontal line as seperator.

    :return: The created seperator
    """
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setLineWidth(1)
    return separator


def get_vertical_separator():
    """
    Creates a simple vertical line as seperator.

    :return: The created seperator
    """
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.VLine)
    separator.setLineWidth(1)
    return separator


def get_button(text):
    """
    Creates a button and does the styling etc.
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
        icon = QIcon(pixmap)
        box.addItem(icon, '')
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
            item = layout.takeAt(layout.count()-1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clear_layout(item.layout())


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ViNo - Visualisation of Norms")
        self.setWindowIcon(QtGui.QIcon(os.path.join(basedir,'logo_vino.ico')))
        self.setMinimumSize(1000, 600)
        self.norm_system = None

        self.info_widget = InfoWidget(norm_system=self.norm_system, parent=self)
        self.edit_widget = EditWidget(norm_system=self.norm_system, parent=self)
        self.premise_aspect = AspectWidget(parent=self, aspect_name=NormSystem.aspect_premise, norm_system=self.norm_system, display_type=display_interactive)
        self.domain_aspect = AspectWidget(parent=self, aspect_name=NormSystem.aspect_domain, norm_system=self.norm_system, display_type=display_interactive)

        layout_main = QHBoxLayout()
        layout_main.addWidget(self.info_widget, 2)

        self.tab_container = QTabWidget()
        self.tab_container.addTab(self.edit_widget, 'Edit Data')
        self.tab_container.addTab(self.premise_aspect, self.premise_aspect.aspect_name)
        self.tab_container.addTab(self.domain_aspect, self.domain_aspect.aspect_name)
        layout_main.addWidget(self.tab_container, 5)

        widget = QWidget()
        widget.setLayout(layout_main)
        self.setCentralWidget(widget)

    def update_highlighted_case(self, case):
        self.edit_widget.update_highlighted_case(case)

    def update_highlighted_norm(self, norm):
        self.edit_widget.update_highlighted_norm(norm)

    def update_display_type(self, t):
        self.domain_aspect.set_display_type(t)
        self.premise_aspect.set_display_type(t)
        self.update_plotting()

    def set_display_type(self, t):
        self.info_widget.set_display_type(t)
        self.update_display_type(t)

    def update_cases(self):
        self.info_widget.update_cases()

    def update_norms(self):
        self.info_widget.update_norms()

    def update_norm_system(self, norm_system):
        self.info_widget.set_norm_system(norm_system)
        self.domain_aspect.set_norm_system(norm_system)
        self.premise_aspect.set_norm_system(norm_system)

    def update_plotting(self):
        self.premise_aspect.plot_current_setup()
        self.domain_aspect.plot_current_setup()


class InfoWidget(QWidget):
    """
    Left side of the GUI. Displays created Norms and Cases and some general settings
    """
    def __init__(self, parent, norm_system, *args, **kwargs):
        super().__init__( parent=parent, *args, **kwargs)
        self.parent=weakref.proxy(parent)
        self.norm_system = norm_system
        self.norm_list = QVBoxLayout()
        self.case_list = QVBoxLayout()
        self.norm_scroll = QScrollArea()
        self.case_scroll = QScrollArea()
        self.colors_norms_type = QComboBox()
        self.colors_hex_input = QLineEdit()
        self.cases_highlight_group = HighlightGroup()
        self.norms_highlight_group = HighlightGroup()

        layout = QVBoxLayout()

        norm_system_layout = QVBoxLayout()
        norm_system_layout.addWidget(get_title_label('Norm System'))
        #display type
        display_input = QHBoxLayout()
        display_input.addWidget(get_normal_label('Select display type:'))
        self.display_combo = QComboBox()
        self.display_combo.addItems(display_types)
        self.display_combo.setCurrentText(display_interactive)
        self.display_combo.currentTextChanged.connect(self.update_display_type)
        display_input.addWidget(self.display_combo)
        norm_system_layout.addLayout(display_input)

        norm_system_layout.addWidget(get_horizontal_separator())

        # change Norm color
        color_settings_layout = QVBoxLayout()
        heading_layout = QHBoxLayout()
        heading_layout.addWidget(get_normal_label('Change color of norm_aspect_value type:'), 1)
        setup_input_field(line_edit=self.colors_hex_input, validator_regex='^#(?:[0-9a-fA-F]{6})$',
                          placeholder_text='Hex color code')
        color_settings_layout.addLayout(heading_layout)
        color_settings_layout.addWidget(self.colors_norms_type, 1)

        selection_layout = QHBoxLayout()
        selection_layout.addWidget(self.colors_hex_input, 1)
        change_button = get_button('Set Color')
        change_button.clicked.connect(self.change_one_color)
        selection_layout.addWidget(change_button, 1)
        color_settings_layout.addLayout(selection_layout)
        norm_system_layout.addLayout(color_settings_layout)

        norm_system_layout.addWidget(get_horizontal_separator())

        # reset colors
        reset_colors_button = get_button('Reset all colors')
        reset_colors_button.clicked.connect(self.reset_colors)
        norm_system_layout.addWidget(reset_colors_button)

        layout.addLayout(norm_system_layout)

        # norm_aspect_value list
        norms_layout = QVBoxLayout()
        norms_layout.addWidget(get_title_label('Norms'))

        norm_list_widget = QWidget()
        norm_list_widget.setLayout(self.norm_list)
        self.norm_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.norm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.norm_scroll.setWidgetResizable(True)
        self.norm_list.addStretch()
        self.norm_scroll.setWidget(norm_list_widget)
        norms_layout.addWidget(self.norm_scroll)
        layout.addLayout(norms_layout)

        # cases list
        cases_layout = QVBoxLayout()
        cases_layout.addWidget(get_title_label('Cases'))
        case_list_widget = QWidget()
        case_list_widget.setLayout(self.case_list)
        self.case_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.case_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.case_scroll.setWidgetResizable(True)
        self.case_list.addStretch()
        self.case_scroll.setWidget(case_list_widget)
        cases_layout.addWidget(self.case_scroll)
        layout.addLayout(cases_layout)

        self.setLayout(layout)

    def set_display_type(self, display_type):
        self.display_combo.setCurrentText(display_type)

    def update_display_type(self, t):
        self.parent.update_display_type(t)

    def change_one_color(self):
        """
        Sets a new color to one of the norm_aspect_value types
        """
        if self.norm_system is not None:
            aspect_name = None
            for aspect_n in self.norm_system.aspect_name_list:
                if self.colors_norms_type.currentText() in self.norm_system.aspects[aspect_n].type_names:
                    aspect_name=aspect_n
                    break
            if aspect_name is not None:
                self.norm_system.change_color(self.colors_norms_type.currentText(), self.colors_hex_input.text(),
                                          aspect_name)
                self.parent.update_plotting()

    def reset_colors(self):
        if self.norm_system is not None:
            for aspect in self.norm_system.aspects.keys():
                self.norm_system.reset_colors(aspect)
            self.parent.update_plotting()

    def set_norm_system(self, norm_system):
        self.norm_system=norm_system
        for aspect_name in norm_system.aspects.keys():
            self.colors_norms_type.addItems(norm_system.aspects[aspect_name].type_names)

    def update_cases(self):
        clear_layout(self.case_list)
        self.cases_highlight_group.clear()
        self.case_list.addStretch()
        cases = set(elem for key in NormSystem.aspect_name_list for elem in self.norm_system.case_list[key])
        cases = sorted(cases, key=lambda one_case: one_case.identifier, reverse=True)
        for case in cases:
            premise_domain = ', '.join(case.case_aspect_values.keys())
            latex_expression = '$ \\mathbf{' + case.identifier.replace('$', '') + '}: ('+premise_domain+')$'
            pix_map = latex_to_qpixmap(latex_expression)
            case_label = HighlightableButton(click_event=lambda x: self.parent.update_highlighted_case(x), data=case, pixmap=pix_map, highlightgroup=self.cases_highlight_group)
            self.case_list.insertWidget(0, case_label)

    def update_norms(self):
        """
        Updates the list showing the norms. Removes the old entries and inserts current norms.
        """
        clear_layout(self.norm_list)
        self.norms_highlight_group.clear()
        self.norm_list.addStretch()
        norms = self.norm_system.norm_list
        for norm in norms:
            widget = QWidget()
            h_layout = QHBoxLayout()

            premise_list = [aspect_value.identifier.replace('$','') for aspect_value in norm.norm_aspect_values[NormSystem.aspect_premise]]
            premise_list.sort()
            premise_domain = ', '.join(premise_list)
            if premise_domain == '':
                premise_domain = '-'
            latex_expression = '$ \\mathbf{' + norm.domain_aspect().identifier + '}: '+premise_domain+'$'
            pix_map = latex_to_qpixmap(latex_expression)
            norm_label = HighlightableButton(click_event=lambda x: self.parent.update_highlighted_norm(x), data=norm, highlightgroup=self.norms_highlight_group, pixmap=pix_map)
            h_layout.addWidget(norm_label)

            h_layout.setContentsMargins(0, 0, 0, 0)
            widget.setLayout(h_layout)
            self.norm_list.insertWidget(self.norm_list.count() - 1, widget)


class EditWidget(QWidget):
    """
    Widget for editing all data (a Tab on the right)
    """
    def __init__(self,norm_system, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.parent=weakref.proxy(parent)
        self.norm_system=norm_system
        self.pixmap = {}
        self.case_input = CaseInputWidget(norm_system=self.norm_system, parent=self)
        self.domain_norm_input = DomainNormInputWidget(norm_system=self.norm_system, parent=self)
        self.premise_norm_input = PremiseNormInputWidget(norm_system=self.norm_system, parent=self)

        layout = QVBoxLayout()
        layout.addWidget(get_title_label('Edit Data'))

        self.toolbox = QToolBox()
        self.toolbox.addItem(NormSystemInputWidget(norm_system=self.norm_system, parent=self), "Norm System")
        self.toolbox.addItem(self.case_input, "Case")
        self.toolbox.addItem(self.premise_norm_input, "Norm - "+NormSystem.aspect_premise)
        self.toolbox.addItem(self.domain_norm_input, "Norm - "+NormSystem.aspect_domain)

        self.toolbox.setItemEnabled(1, False)
        self.toolbox.setItemEnabled(2, False)
        self.toolbox.setItemEnabled(3, False)
        layout.addWidget(self.toolbox)

        self.setLayout(layout)

    def update_highlighted_case(self,case):
        self.case_input.update_highlighted_case(case)

    def update_highlighted_norm(self,norm):
        self.domain_norm_input.update_highlighted_norm(norm)
        self.premise_norm_input.update_highlighted_domain_norm(norm)

    def update_plotting(self):
        self.parent.update_plotting()

    def update_norm_system(self, norm_system):
        # only calles from NormSystemInputWidget
        self.norm_system=norm_system
        self.parent.update_norm_system(norm_system)

        for key in NormSystem.dimensions+[NormSystem.hierarchy]:
            for aspect_key in self.norm_system.aspects.keys():
                set_pixmap(self.pixmap, self.norm_system.aspects[aspect_key].scales[key])

        # contact the remaining children
        self.toolbox.setItemEnabled(1, True)
        self.toolbox.setItemEnabled(2, True)
        self.toolbox.setItemEnabled(3, True)
        self.case_input.update_norm_system(norm_system)
        self.domain_norm_input.update_norm_system(norm_system)
        self.premise_norm_input.update_norm_system(norm_system)

    def update_cases(self):
        self.parent.update_cases()

    def update_norms(self):
        self.parent.update_norms()
        self.premise_norm_input.update_domain_norms()


class DomainNormInputWidget(QWidget):
    """
    Widget for entering information about the domain aspect. Expandable box in the EditWidget
    """
    start_str = 'start'
    end_str = 'end'
    def __init__(self, norm_system, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.norm_system = norm_system
        self.parent = weakref.proxy(parent)
        self.hierarchy_combo = QComboBox()
        self.combos = {dim:{self.start_str:QComboBox(), self.end_str:QComboBox()} for dim in NormSystem.dimensions}
        self.aspect_types = QComboBox()
        self.starttime = QLineEdit()
        self.identifier = QLineEdit()
        self.input_error = QLabel()
        self.highlighted_norm = None

        layout = QVBoxLayout()

        for dim in NormSystem.dimensions:
            input_line = QHBoxLayout()
            input_line.addWidget(get_one_letter_normal_label(dim))
            input_line.addWidget(self.combos[dim][self.start_str],1)
            input_line.addWidget(self.combos[dim][self.end_str],1)
            layout.addLayout(input_line)

        type_layout = QHBoxLayout()
        type_layout.addWidget(get_one_letter_normal_label(NormSystem.hierarchy))
        type_layout.addWidget(self.hierarchy_combo,1)
        type_layout.addWidget(self.aspect_types,1)
        layout.addLayout(type_layout)

        id_input = QHBoxLayout()
        id_input.addWidget(get_one_letter_normal_label(' '))
        self.starttime.setPlaceholderText('Temporal indicator')
        id_input.addWidget(self.starttime)
        self.identifier.setPlaceholderText('Identifier')
        id_input.addWidget(self.identifier)
        layout.addLayout(id_input)

        self.button_container = QHBoxLayout()
        layout.addLayout(self.button_container)
        layout.addWidget(self.input_error)
        self.input_error.setFixedHeight(0)

        self.setLayout(layout)
        self.update_highlighted_norm(None)

    def update_highlighted_norm(self, norm):
        clear_layout(self.button_container)
        self.highlighted_norm = norm
        if norm is None:
            add_norm_button = get_button('Add Norm')
            add_norm_button.clicked.connect(self.add_norm)
            self.button_container.addWidget(add_norm_button)
        else:
            button_edit_norm = get_button('Edit this Norm')
            button_delete_norm = get_button('Delete this Norm')
            button_delete_norm.clicked.connect(self.delete_norm)
            button_edit_norm.clicked.connect(self.edit_norm)
            self.button_container.addWidget(button_edit_norm, 1)
            self.button_container.addWidget(button_delete_norm, 1)

            aspect_values = norm.domain_aspect()
            self.identifier.setText(aspect_values.identifier)
            self.starttime.setText(aspect_values.starttime)
            for dim in NormSystem.dimensions:
                self.combos[dim][self.start_str].setCurrentIndex(aspect_values.start_values[dim].value)
                self.combos[dim][self.end_str].setCurrentIndex(aspect_values.end_values[dim].value)
            self.hierarchy_combo.setCurrentIndex(aspect_values.hierarchy.value)

    def update_norm_system(self, norm_system):
        self.norm_system=norm_system

        # norm_aspect_value types
        self.aspect_types.clear()
        type_list = list(self.norm_system.aspects[NormSystem.aspect_domain].type_names)
        type_list.sort()
        self.aspect_types.addItems(type_list)

        for dim_key in NormSystem.dimensions:
            # dim combos
            reset_combo_box(self.combos[dim_key][self.start_str],
                            self.parent.pixmap,
                            self.norm_system.aspects[NormSystem.aspect_domain].scales[dim_key])
            reset_combo_box(self.combos[dim_key][self.end_str],
                            self.parent.pixmap,
                            self.norm_system.aspects[NormSystem.aspect_domain].scales[dim_key])

        reset_combo_box(self.hierarchy_combo, self.parent.pixmap,
                        self.norm_system.aspects[NormSystem.aspect_domain].scales[NormSystem.hierarchy])

    def edit_norm(self):
        self.input_error.setText(' ')
        self.input_error.setFixedHeight(0)
        self.norm_system.delete_domain_norm(self.highlighted_norm)
        self.add_norm()
        self.update_highlighted_norm(None)

    def delete_norm(self):
        self.norm_system.delete_domain_norm(self.highlighted_norm)
        self.parent.update_norms()
        self.parent.update_plotting()
        self.update_highlighted_norm(None)

    def add_norm(self):
        """
        Adds a norm_aspect_value if the entered values are correct. Displays the errors otherwise.
        """
        errortext = []
        norm_id = self.identifier.text()
        if norm_id.strip() == '':
            errortext.append('Identifier must be given.')
        starttime = self.starttime.text()
        if starttime.strip() == '':
            errortext.append('Temporal indicator must be given.')
        if len(errortext) > 0:
            self.input_error.setText('\n'.join(errortext))
            self.input_error.setFixedHeight(len(errortext) * one_row_normal_label_height)
        else:
            self.input_error.setText('')
            self.input_error.setFixedHeight(0)
            vals = {dim:{self.start_str:self.combos[dim][self.start_str].currentData(Qt.ItemDataRole.UserRole),
                                     self.end_str:self.combos[dim][self.end_str].currentData(Qt.ItemDataRole.UserRole)} for dim in NormSystem.dimensions}
            vals = {dim:{self.start_str:DimValue(name=vals[dim][self.start_str],
                                                             value=self.norm_system.get_position(vals[dim][self.start_str], dim, NormSystem.aspect_domain)),
                                     self.end_str:DimValue(name=vals[dim][self.end_str],
                                                             value=self.norm_system.get_position(vals[dim][self.end_str], dim, NormSystem.aspect_domain))}
                                for dim in NormSystem.dimensions}
            h_val = self.hierarchy_combo.currentData(Qt.ItemDataRole.UserRole)
            domain_aspect = NormAspectValues(
                                          o_vals=(vals[NormSystem.dim_o][self.start_str],
                                                  vals[NormSystem.dim_o][self.end_str]),
                                          r_vals=(vals[NormSystem.dim_r][self.start_str],
                                                  vals[NormSystem.dim_r][self.end_str]),
                                          s_vals=(vals[NormSystem.dim_s][self.start_str],
                                                  vals[NormSystem.dim_s][self.end_str]),
                                          t_vals=(vals[NormSystem.dim_t][self.start_str],
                                                  vals[NormSystem.dim_t][self.end_str]),
                                          norm_type=self.aspect_types.currentText(),
                                          aspect_name=NormSystem.aspect_domain,
                                          identifier=norm_id, starttime=starttime,
                                          h_val=DimValue(name=h_val, value=self.norm_system.get_position(h_val, NormSystem.hierarchy, NormSystem.aspect_domain)))
            self.norm_system.add_norm(Norm(domain_aspect=domain_aspect))
            self.parent.update_norms()
            self.parent.update_plotting()


class PremiseNormInputWidget(QWidget):
    """
    Widget for entering information about the premise aspect. Expandable box in the EditWidget
    """
    start_str = 'start'
    end_str = 'end'
    def __init__(self, norm_system, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.norm_system = norm_system
        self.parent = weakref.proxy(parent)
        self.hierarchy_combo = QComboBox()
        self.combos = {dim:{self.start_str:QComboBox(), self.end_str:QComboBox()} for dim in NormSystem.dimensions}
        self.aspect_types = QComboBox()
        self.starttime = QLineEdit()
        self.input_error = QLabel()
        self.highlighted_domain_norm = None
        self.highlighted_premise_aspect_values = None
        self.norm_combo = QComboBox()
        self.premise_norm_list = QVBoxLayout()
        self.premise_norm_scroll = QScrollArea()
        self.norm_pixmaps = {}
        self.premise_highlight_group = HighlightGroup()

        layout = QHBoxLayout()
        input_layout = QVBoxLayout()

        for dim in NormSystem.dimensions:
            input_line = QHBoxLayout()
            input_line.addWidget(get_one_letter_normal_label(dim))
            input_line.addWidget(self.combos[dim][self.start_str],1)
            input_line.addWidget(self.combos[dim][self.end_str],1)
            input_layout.addLayout(input_line)

        type_layout = QHBoxLayout()
        type_layout.addWidget(get_one_letter_normal_label(NormSystem.hierarchy))
        type_layout.addWidget(self.hierarchy_combo,1)
        type_layout.addWidget(self.aspect_types,1)
        input_layout.addLayout(type_layout)

        id_input = QHBoxLayout()
        id_input.addWidget(get_one_letter_normal_label(' '))
        self.starttime.setPlaceholderText('Temporal indicator')
        id_input.addWidget(self.starttime)
        input_layout.addLayout(id_input)

        self.button_container = QHBoxLayout()
        input_layout.addLayout(self.button_container)
        input_layout.addWidget(self.input_error)
        self.input_error.setFixedHeight(0)

        select_layout = QVBoxLayout()
        select_layout.addWidget(get_normal_label('Select corresponding '+NormSystem.aspect_domain+' norm:'))
        select_layout.addWidget(self.norm_combo)
        self.norm_combo.currentIndexChanged.connect(self.update_premise_norm_list)

        select_layout.addWidget(get_normal_label('Existing '+NormSystem.aspect_premise+' norms:'))

        norm_list_widget = QWidget()
        norm_list_widget.setLayout(self.premise_norm_list)
        self.premise_norm_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.premise_norm_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.premise_norm_scroll.setWidgetResizable(True)
        self.premise_norm_list.addStretch()
        self.premise_norm_scroll.setWidget(norm_list_widget)
        select_layout.addWidget(self.premise_norm_scroll)

        layout.addLayout(input_layout,2)
        layout.addWidget(get_vertical_separator())
        layout.addLayout(select_layout,1)
        self.setLayout(layout)
        self.update_highlighted_premise_norm(None)

    def update_premise_norm_list(self):
        clear_layout(self.premise_norm_list)
        self.premise_highlight_group.clear()
        self.premise_norm_list.addStretch()

        norm_id = self.norm_combo.currentData(Qt.ItemDataRole.UserRole)
        norm = self.norm_system.get_norm(norm_id)
        if norm is not None:
            for aspect_value in norm.norm_aspect_values[NormSystem.aspect_premise]:
                pix_map = latex_to_qpixmap(aspect_value.identifier)
                norm_label = HighlightableButton(click_event=lambda x: self.update_highlighted_premise_norm(x), data=aspect_value,
                                                 highlightgroup=self.premise_highlight_group, pixmap=pix_map)
                self.premise_norm_list.insertWidget(self.premise_norm_list.count() - 1, norm_label)

    def update_highlighted_domain_norm(self, norm):
        self.highlighted_domain_norm = norm
        if norm is not None:
            domain_aspect = norm.domain_aspect()
            index = self.norm_combo.findData(domain_aspect.identifier, Qt.ItemDataRole.UserRole)
            self.norm_combo.setCurrentIndex(index)

    def update_highlighted_premise_norm(self, aspect_values):
        clear_layout(self.button_container)
        self.highlighted_premise_aspect_values = aspect_values
        if aspect_values is None:
            add_norm_button = get_button('Add Norm')
            add_norm_button.clicked.connect(self.add_norm)
            self.button_container.addWidget(add_norm_button)
        else:
            button_edit_norm = get_button('Edit this Norm')
            button_delete_norm = get_button('Delete this Norm')
            button_delete_norm.clicked.connect(self.delete_norm)
            button_edit_norm.clicked.connect(self.edit_norm)
            self.button_container.addWidget(button_edit_norm, 1)
            self.button_container.addWidget(button_delete_norm, 1)

            self.starttime.setText(aspect_values.starttime)
            for dim in NormSystem.dimensions:
                self.combos[dim][self.start_str].setCurrentIndex(aspect_values.start_values[dim].value)
                self.combos[dim][self.end_str].setCurrentIndex(aspect_values.end_values[dim].value)
            self.hierarchy_combo.setCurrentIndex(aspect_values.hierarchy.value)

    def update_domain_norms(self):
        norms = self.norm_system.norm_list
        identifier_list = []
        for norm in norms:
            domain_aspect_values = norm.domain_aspect()
            identifier_list.append(domain_aspect_values.identifier)
            if domain_aspect_values.identifier not in self.norm_pixmaps.keys():
                latex_expression = '$' + domain_aspect_values.identifier + '$'
                self.norm_pixmaps[norm.domain_aspect().identifier] = latex_to_qpixmap(latex_expression)

        reset_combo_box(box=self.norm_combo, pix_map=self.norm_pixmaps, values=identifier_list)
        self.update_premise_norm_list()

    def update_norm_system(self, norm_system):
        self.norm_system=norm_system

        # norm_aspect_value types
        self.aspect_types.clear()
        type_list = list(self.norm_system.aspects[NormSystem.aspect_premise].type_names)
        type_list.sort(reverse=True)
        self.aspect_types.addItems(type_list)

        for dim_key in NormSystem.dimensions:
            # dim combos
            reset_combo_box(self.combos[dim_key][self.start_str],
                            self.parent.pixmap,
                            self.norm_system.aspects[NormSystem.aspect_premise].scales[dim_key])
            reset_combo_box(self.combos[dim_key][self.end_str],
                            self.parent.pixmap,
                            self.norm_system.aspects[NormSystem.aspect_premise].scales[dim_key])

        reset_combo_box(self.hierarchy_combo, self.parent.pixmap,
                        self.norm_system.aspects[NormSystem.aspect_premise].scales[NormSystem.hierarchy])

    def edit_norm(self):
        abort_reason = self.norm_system.delete_norm_premise_aspect(self.highlighted_premise_aspect_values)
        if abort_reason == '':
            self.input_error.setText(' ')
            self.input_error.setFixedHeight(0)
            self.add_norm()
            self.update_highlighted_premise_norm(None)
        else:
            self.input_error.setText(abort_reason)
            self.input_error.setFixedHeight(one_row_normal_label_height)

    def delete_norm(self):
        abort_reason = self.norm_system.delete_norm_premise_aspect(self.highlighted_premise_aspect_values)
        if abort_reason == '':
            self.input_error.setText(' ')
            self.input_error.setFixedHeight(0)
            self.parent.update_norms()
            self.parent.update_plotting()
            self.update_highlighted_premise_norm(None)
        else:
            self.input_error.setText(abort_reason)
            self.input_error.setFixedHeight(one_row_normal_label_height)

    def add_norm(self):
        """
        Adds a norm_aspect_value if the entered values are correct. Displays the errors otherwise.
        """
        errortext = []
        norm_id = self.norm_combo.currentData(Qt.ItemDataRole.UserRole)
        if norm_id is None:
            errortext.append('A corresponding norm must be given.')
        starttime = self.starttime.text()
        if starttime.strip() == '':
            errortext.append('Temporal indicator must be given.')
        if len(errortext) > 0:
            self.input_error.setText('\n'.join(errortext))
            self.input_error.setFixedHeight(len(errortext) * one_row_normal_label_height)
        else:
            self.input_error.setText('')
            self.input_error.setFixedHeight(0)
            vals = {dim:{self.start_str:self.combos[dim][self.start_str].currentData(Qt.ItemDataRole.UserRole),
                                     self.end_str:self.combos[dim][self.end_str].currentData(Qt.ItemDataRole.UserRole)} for dim in NormSystem.dimensions}
            vals = {dim:{self.start_str:DimValue(name=vals[dim][self.start_str],
                                                             value=self.norm_system.get_position(vals[dim][self.start_str], dim, NormSystem.aspect_premise)),
                                     self.end_str:DimValue(name=vals[dim][self.end_str],
                                                             value=self.norm_system.get_position(vals[dim][self.end_str], dim, NormSystem.aspect_premise))}
                                for dim in NormSystem.dimensions}
            h_val = self.hierarchy_combo.currentData(Qt.ItemDataRole.UserRole)
            premise_aspect = NormAspectValues(
                                          o_vals=(vals[NormSystem.dim_o][self.start_str],
                                                  vals[NormSystem.dim_o][self.end_str]),
                                          r_vals=(vals[NormSystem.dim_r][self.start_str],
                                                  vals[NormSystem.dim_r][self.end_str]),
                                          s_vals=(vals[NormSystem.dim_s][self.start_str],
                                                  vals[NormSystem.dim_s][self.end_str]),
                                          t_vals=(vals[NormSystem.dim_t][self.start_str],
                                                  vals[NormSystem.dim_t][self.end_str]),
                                          norm_type=self.aspect_types.currentText(),
                                          aspect_name=NormSystem.aspect_premise,
                                          identifier=norm_id, starttime=starttime,
                                          h_val=DimValue(name=h_val, value=self.norm_system.get_position(h_val, NormSystem.hierarchy, NormSystem.aspect_premise)))
            abort_reason = self.norm_system.add_norm_premise_aspect(norm_identifier=norm_id,
                                                     premise_aspect=premise_aspect)
            if abort_reason != '':
                self.input_error.setText(abort_reason)
                self.input_error.setFixedHeight(one_row_normal_label_height)
            else:
                self.parent.update_norms()
                self.parent.update_plotting()

class CaseInputWidget(QWidget):
    """
    Widget for entering information about cases. Expandable box in the EditWidget
    """
    def __init__(self, norm_system, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.norm_system = norm_system
        self.parent = weakref.proxy(parent)
        self.name_edit = QLineEdit()
        self.combos = {key:{ NormSystem.dim_o:QComboBox(), NormSystem.dim_r:QComboBox(), NormSystem.dim_s:QComboBox(),
                             NormSystem.dim_t:QComboBox()} for key in NormSystem.aspect_name_list}
        self.dont_use_checkbox = QCheckBox('Don\'t use '+NormSystem.aspect_premise)
        self.error_label = QLabel()
        self.highlighted_case = None
        self.aspect_keys = []
        layout = QVBoxLayout()

        for aspect_name in NormSystem.aspect_name_list:
            aspect_layout = QVBoxLayout()
            heading = QHBoxLayout()
            heading.addWidget(get_normal_label(aspect_name),1)
            if aspect_name == NormSystem.aspect_premise:
                heading.addWidget(self.dont_use_checkbox,1)

            aspect_layout.addLayout(heading)
            for key in self.combos[aspect_name].keys():
                aspect_layout.addWidget(self.combos[aspect_name][key])
            layout.addLayout(aspect_layout, 1)

            layout.addWidget(get_horizontal_separator())

        self.dont_use_checkbox.stateChanged.connect(self.activate_premise)

        self.name_edit.setPlaceholderText('Identifier')
        layout.addWidget(self.name_edit)

        self.button_container = QHBoxLayout()
        layout.addLayout(self.button_container)

        layout.addWidget(self.error_label)
        self.error_label.setFixedHeight(one_row_normal_label_height)
        self.setLayout(layout)
        self.update_highlighted_case(None)

    def update_highlighted_case(self, case):
        clear_layout(self.button_container)
        self.highlighted_case = case
        if case is None:
            button_add_case = get_button('Add Case')
            button_add_case.clicked.connect(self.add_case)
            self.button_container.addWidget(button_add_case)
        else:
            button_edit_case = get_button('Edit this Case')
            button_delete_case = get_button('Delete this Case')
            button_delete_case.clicked.connect(self.delete_case)
            button_edit_case.clicked.connect(self.edit_case)
            self.button_container.addWidget(button_edit_case,1)
            self.button_container.addWidget(button_delete_case,1)

            self.name_edit.setText(case.identifier)
            for dim in NormSystem.dimensions:
                for aspect_name in NormSystem.aspect_name_list:
                    if aspect_name == NormSystem.aspect_domain:
                        aspect_values = case.case_aspect_values[aspect_name]
                        self.combos[aspect_name][dim].setCurrentIndex(aspect_values.coordinates[dim].value)
                    else:
                        if aspect_name in case.case_aspect_values.keys():#
                            aspect_values = case.case_aspect_values[aspect_name]
                            self.dont_use_checkbox.setChecked(False)
                            self.combos[aspect_name][dim].setCurrentIndex(aspect_values.coordinates[dim].value)
                        else:
                            self.dont_use_checkbox.setChecked(True)

    def permanent_deactivate_premise(self, value):
        self.dont_use_checkbox.setChecked(value)
        self.dont_use_checkbox.setDisabled(value)

    def activate_premise(self, state):
        for combo in self.combos[NormSystem.aspect_premise].values():
            combo.setDisabled(state)

    def update_norm_system(self, norm_system):
        self.norm_system=norm_system
        for key in NormSystem.dimensions:
            self.aspect_keys = [NormSystem.aspect_domain]
            prem_h_scale = self.norm_system.aspects[NormSystem.aspect_premise].scales[NormSystem.hierarchy]
            if  len(prem_h_scale) == 1 and prem_h_scale[0]=='': # premise not used
                self.permanent_deactivate_premise(True)
            else:
                self.permanent_deactivate_premise(False)
                self.aspect_keys.append(NormSystem.aspect_premise)
            for aspect_key in self.aspect_keys:
                reset_combo_box(self.combos[aspect_key][key],
                                self.parent.pixmap,
                                self.norm_system.aspects[aspect_key].scales[key])

    def edit_case(self):
        self.error_label.setText(' ')
        self.norm_system.delete_case(self.highlighted_case)
        self.add_case()
        self.update_highlighted_case(None)

    def delete_case(self):
        self.norm_system.delete_case(self.highlighted_case)
        self.parent.update_cases()
        self.parent.update_plotting()
        self.update_highlighted_case(None)

    def add_case(self):
        """
        Adds a case to the norm_aspect_value system. If no identifier is given, a standard name is used.
        """
        self.error_label.setText(' ')
        case_name = self.name_edit.text()
        if case_name.strip() == '':
            case_name = None

        vals = {aspect_key:{dim:DimValue(name=self.combos[aspect_key][dim].currentData(Qt.ItemDataRole.UserRole),
                                         value=self.norm_system.get_position(
                                             value=self.combos[aspect_key][dim].currentData(Qt.ItemDataRole.UserRole),
                                             dimension=dim, aspect_name=aspect_key))
                     for dim in NormSystem.dimensions} for aspect_key in self.aspect_keys}
        domain_aspect_values = CaseAspectValues(o_val=vals[NormSystem.aspect_domain][NormSystem.dim_o],
                                              r_val=vals[NormSystem.aspect_domain][NormSystem.dim_r],
                                              s_val=vals[NormSystem.aspect_domain][NormSystem.dim_s],
                                              t_val=vals[NormSystem.aspect_domain][NormSystem.dim_o],
                                              aspect_name=NormSystem.aspect_domain)
        premise_aspect_values = None
        if not self.dont_use_checkbox.isChecked():
            premise_aspect_values = CaseAspectValues(o_val=vals[NormSystem.aspect_premise][NormSystem.dim_o],
                                                  r_val=vals[NormSystem.aspect_premise][NormSystem.dim_r],
                                                  s_val=vals[NormSystem.aspect_premise][NormSystem.dim_s],
                                                  t_val=vals[NormSystem.aspect_premise][NormSystem.dim_t],
                                                  aspect_name=NormSystem.aspect_premise)
        self.norm_system.add_case(domain_aspect=domain_aspect_values,
                             premise_aspect= premise_aspect_values, identifier=case_name)

        self.parent.update_cases()
        self.parent.update_plotting()


class NormSystemInputWidget(QWidget):
    """
    Widget for entering information about the NormSystem. Expandable box in the EditWidget
    """
    def __init__(self, norm_system, parent, *args, **kwargs):
        super().__init__(parent=parent,*args, **kwargs)
        self.norm_system=norm_system
        self.parent = weakref.proxy(parent)
        self.dim_raw_texts = {key:{NormSystem.dim_o: '', NormSystem.dim_r: '', NormSystem.dim_s: '',
                              NormSystem.dim_t: '', NormSystem.hierarchy:''} for
                              key in NormSystem.aspect_name_list}
        self.norm_system_error = get_normal_label('')
        self.norm_system_error.setFixedHeight(0)
        self.inputs = {key:{ NormSystem.dim_o:QLineEdit(), NormSystem.dim_r:QLineEdit(), NormSystem.dim_s:QLineEdit(),
                             NormSystem.dim_t:QLineEdit(), NormSystem.hierarchy:QLineEdit()} for key in NormSystem.aspect_name_list}
        self.dont_use_checkbox = QCheckBox('Don\'t use '+NormSystem.aspect_premise)
        self.pix_update = None

        layout = QVBoxLayout()

        for aspect_name in NormSystem.aspect_name_list:
            aspect_layout = QVBoxLayout()
            heading = QHBoxLayout()
            heading.addWidget(get_normal_label(aspect_name))
            if aspect_name == NormSystem.aspect_premise:
                heading.addWidget(self.dont_use_checkbox)
            aspect_layout.addLayout(heading)
            for key in self.inputs[aspect_name].keys():
                aspect_layout.addLayout(self.set_up_norm_system_input_field(placeholder_text=display_names[key],
                                                                         dim_id=key, aspect_name=aspect_name,
                                                                         line_edit=self.inputs[aspect_name][key]))
                self.inputs[aspect_name][key].editingFinished.connect(lambda : self.start_pix_updater(self.inputs[aspect_name][key].text()))
            layout.addLayout(aspect_layout,1)
            layout.addWidget(get_horizontal_separator())

        self.dont_use_checkbox.stateChanged.connect(self.activate_premise)

        button_layout = QHBoxLayout()

        button_dummy_data = get_button('Insert Dummy Data')
        button_dummy_data.clicked.connect(self.insert_dummy_norm_sys)
        button_layout.addWidget(button_dummy_data)

        self.button_container = QHBoxLayout()
        self.button_set_norm_system = get_button('Set Values')
        self.button_set_norm_system.clicked.connect(self.create_norm_system)

        dummy_scales = [get_generic_dim_scale(dim=key, num_entries= random.randrange(2, 7)) for key in NormSystem.dimensions+[NormSystem.hierarchy]]
        self.pix_update = PixmapUpdater(pixmap=self.parent.pixmap, new_values=[x for xs in dummy_scales for x in xs])
        self.pix_update.start()
        self.button_set_norm_system.setEnabled(False)
        self.pix_update.finished.connect(
            lambda: self.button_set_norm_system.setEnabled(True)
        )

        self.button_container.addWidget(self.button_set_norm_system)
        button_layout.addLayout(self.button_container)
        layout.addLayout(button_layout)
        layout.addWidget(self.norm_system_error)
        self.setLayout(layout)



    def start_pix_updater(self, text):
        scale = text.split(', ')
        self.pix_update = PixmapUpdater(pixmap=self.parent.pixmap, new_values=scale)
        self.pix_update.start()

        self.button_set_norm_system.setEnabled(False)
        self.pix_update.finished.connect(
            lambda: self.button_set_norm_system.setEnabled(True)
        )

    def activate_premise(self, state):
        for combo in self.inputs[NormSystem.aspect_premise].values():
            combo.setDisabled(state)

    def create_norm_system(self):
        """
        Creates a NormSystem, if the given values are correct. Otherwise, displays an error.
        """
        all_errors = []
        prem_dims = [['']] * 5
        domain_dims = [['']] * 5
        dim_list = NormSystem.dimensions+[NormSystem.hierarchy]
        for i in range(len(dim_list)):
            scale, errors = self.derive_scale(dim_list[i], aspect_name=NormSystem.aspect_domain)
            all_errors.extend(errors)
            domain_dims[i]=scale
            if not self.dont_use_checkbox.isChecked():
                scale, errors = self.derive_scale(dim_list[i], aspect_name=NormSystem.aspect_premise)
                all_errors.extend(errors)
                prem_dims[i]=scale

        if len(all_errors) > 0:
            self.norm_system_error.setText('\n'.join(all_errors))
            self.norm_system_error.setFixedHeight(len(all_errors) * one_row_normal_label_height)
        else:
            self.norm_system_error.setText('')
            self.norm_system_error.setFixedHeight(0)
            self.norm_system = NormSystem(domain_scales=domain_dims, premise_scales=prem_dims)
            self.parent.update_norm_system(self.norm_system)
            self.parent.update_cases()
            self.parent.update_norms()
            self.parent.update_plotting()

    def derive_scale(self, dimension, aspect_name):
        """
        Derives the scales from the given raw text by splitting it.

        :param dimension: dimension to derive
        :param aspect_name: the aspect to work with
        :return: scale, error_messages with scale the list of split values anr error_messages contianing possible errors
        """
        error_id = 'Dimension ' + dimension
        scale = self.dim_raw_texts[aspect_name][dimension].split(', ')
        error_id += ', Aspect '+ aspect_name
        scale = ['$'+val.replace(' ','\\ ')+'$' for val in scale if val != '']

        num_entries = len(scale)
        error_messages = []
        if num_entries == 0:
            error_messages.append(error_id + ': No entries given.')
        if len(set(scale)) != num_entries:
            error_messages.append(error_id + ': Duplicate entries are given.')
        return scale, error_messages

    def insert_dummy_norm_sys(self):
        """
        Inserts dummy data in the line edits for creating a norm_aspect_value system.
        """
        for key in self.inputs[NormSystem.aspect_domain].keys():
            if not self.dont_use_checkbox.isChecked():
                self.inputs[NormSystem.aspect_premise][key].setText(', '.join(
                    get_generic_dim_scale(dim=key, num_entries= random.randrange(2, 7)))
                                                 .replace('$', '').replace('{', '').replace('}', ''))
            self.inputs[NormSystem.aspect_domain][key].setText(', '.join(
                get_generic_dim_scale(dim=key, num_entries= random.randrange(2, 7)))
                                             .replace('$', '').replace('{', '').replace('}', ''))

    def set_up_norm_system_input_field(self, placeholder_text, dim_id, line_edit, aspect_name):
        """
        Sets up one input layout for a norm_aspect_value system dimension, which consists of a QLabel for the dimension indicator
        and a QLineEdit for inputting values

        :param placeholder_text: Placeholder Text for the input fields
        :param dim_id: Dimension indicator
        :param line_edit: the line_edit to use
        :param aspect_name: the aspect to work with
        :return: the finished layout
        """
        layout = QHBoxLayout()
        layout.addWidget(get_one_letter_normal_label(dim_id))
        regex_dim_input = "[A-Za-z1-90<=>${_-} ]+(, [A-Za-z1-90<=>${_-} ]+)*"
        setup_input_field(line_edit=line_edit, placeholder_text=placeholder_text, validator_regex=regex_dim_input)
        line_edit.textChanged.connect(lambda state: self.set_dim_raw_text(dim=dim_id, text=state, aspect_name=aspect_name))
        layout.addWidget(line_edit)
        return layout

    def set_dim_raw_text(self, dim, text, aspect_name):
        """
        Sets the raw text for the given dimension

        :param dim: Given dimension
        :param aspect_name: The aspect to wirk with
        :param text: raw text data (to be split into list)
        """
        self.dim_raw_texts[aspect_name][dim] = text


class AspectWidget(QWidget):
    """
    Widget for displaying the figures of one aspect. As a Tab on the right
    """
    display_type = None

    def __init__(self, parent, aspect_name, norm_system, display_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent=weakref.proxy(parent)
        self.aspect_name=aspect_name
        self.display_type=display_type
        self.norm_system=norm_system
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.web_view = QWebEngineView()
        self.web_view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.layout = QVBoxLayout()

        # Dimensions, top
        self.settings = SettingsWidget(parent=self, aspect_name=self.aspect_name, norm_system=self.norm_system)
        self.layout.addWidget(self.settings)

        self.interactive_widget = QWidget()
        interactive_layout = QVBoxLayout()
        interactive_layout.addWidget(self.web_view, 1)
        self.interactive_widget.setLayout(interactive_layout)
        self.web_view.show()

        self.classic_widget = QWidget()
        classic_layout = QVBoxLayout()
        classic_layout.addWidget(self.toolbar)
        classic_layout.addWidget(self.canvas, 1)
        self.classic_widget.setLayout(classic_layout)

        self.layout.addWidget(self.interactive_widget)
        self.setLayout(self.layout)

    def set_display_type(self, t):
        if self.display_type is None or (t != self.display_type and display_interactive in [t, self.display_type]):
            if t == display_interactive:
                self.classic_widget.setVisible(False)
                self.interactive_widget.setVisible(True)
                self.layout.replaceWidget(self.classic_widget, self.interactive_widget)
            else:
                self.interactive_widget.setVisible(False)
                self.classic_widget.setVisible(True)
                self.layout.replaceWidget(self.interactive_widget, self.classic_widget)
        self.display_type = t

    def set_norm_system(self, norm_system):
        self.norm_system = norm_system
        self.settings.set_norm_system(norm_system)

    def plot_current_setup(self):
        """
        Plots the data entered to the NormSystem.
        """
        if self.norm_system is not None:
            dim_x = self.settings.get_x_axis()
            dim_y = self.settings.get_y_axis()
            dim_z = self.settings.get_z_axis()
            if dim_z != none_selected_str and self.display_type != display_interactive:
                self.parent.set_display_type(display_interactive)
            if self.display_type!=display_interactive:
                self.figure.clear()
                if len(self.norm_system.case_list[self.aspect_name])>0:
                    if self.settings.get_show_all_one_dim():
                        self.norm_system.draw_dims_one_all(figure=self.figure, aspect_name=self.aspect_name, display_type=self.display_type)
                    elif (self.settings.get_show_both_two_dims()
                          and dim_y != none_selected_str):
                        dims_one = (dim_x, dim_y)
                        other_dims = [key for key in display_names
                                      if key not in [dim_x, dim_y, NormSystem.hierarchy, none_selected_str]]
                        dims_two = (other_dims[0], other_dims[1])
                        self.norm_system.draw_dims_two_all(figure=self.figure, display_type=self.display_type,
                                                           dims_one=dims_one, dims_two=dims_two,
                                                           aspect_name=self.aspect_name)
                    elif dim_y != none_selected_str:
                        self.norm_system.draw_dims_two(fig=self.figure, display_type=self.display_type,
                                                       dim_x=dim_x, dim_y=dim_y,
                                                       aspect_name=self.aspect_name)
                    else:
                        self.norm_system.draw_dims_one(fig=self.figure, x_dim=dim_x, display_type=self.display_type,
                                                       aspect_name=self.aspect_name)
                    self.canvas.draw()
            else:  # interactive
                if len(self.norm_system.case_list[self.aspect_name])>0:
                    if self.settings.get_show_all_one_dim():
                        web_figure = self.norm_system.draw_dims_one_all(aspect_name=self.aspect_name, display_type=self.display_type)
                    elif (self.settings.get_show_both_two_dims()
                          and dim_y != none_selected_str):
                        dims_one = (dim_x, dim_y)
                        other_dims = [key for key in display_names
                                      if key not in [dim_x, dim_y, NormSystem.hierarchy, none_selected_str]]
                        dims_two = (other_dims[0], other_dims[1])
                        web_figure = self.norm_system.draw_dims_two_all(display_type=self.display_type,
                                                           dims_one=dims_one, dims_two=dims_two,
                                                           aspect_name=self.aspect_name)
                    elif dim_z != none_selected_str:
                        dim_to_exlude = [dim for dim in NormSystem.dimensions if dim not in [dim_x,dim_y, dim_z]][0]
                        web_figure = self.norm_system.draw_dims_three(aspect_name=self.aspect_name, dim_to_exclude=dim_to_exlude)
                    elif dim_y != none_selected_str:
                        web_figure = self.norm_system.draw_dims_two(fig=self.figure, display_type=self.display_type,
                                                       dim_x=dim_x, dim_y=dim_y,
                                                       aspect_name=self.aspect_name)
                    else:
                        web_figure = self.norm_system.draw_dims_one(fig=self.figure, x_dim=dim_x, display_type=self.display_type,
                                                       aspect_name=self.aspect_name)

                    # write to temp file, because setHtml doesnt work
                    temp_file = '.\\tmp_interactive'+self.aspect_name+'.html'
                    web_figure.write_html(temp_file, include_mathjax='cdn')
                    file_path = os.path.abspath(temp_file)
                    self.web_view.load(QUrl.fromLocalFile(file_path))
                else:
                    self.web_view.setHtml('')


class SettingsWidget(QWidget):
    """
    Widget used in the AspectWidget to adjust which dimensions etc. should be displayed
    """
    def __init__(self, aspect_name, norm_system,parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.aspect_name=aspect_name
        self.parent = weakref.proxy(parent)
        self.norm_system=norm_system

        self.x_axis = NormSystem.dim_o
        self.y_axis = none_selected_str
        self.z_axis = none_selected_str
        self.show_all_one_dim = False
        self.show_both_two_dims = False
        self.combo_x_dim = QComboBox()
        self.combo_y_dim = QComboBox()
        self.combo_z_dim = QComboBox()
        self.checkbox_all = QCheckBox('Display other combination')

        layout = QVBoxLayout()
        self.setup_layout(layout)
        self.setLayout(layout)

    def setup_layout(self, layout):
        layout.addWidget(get_title_label('Settings'))
        dim_settings_layout = QHBoxLayout()

        # x axis
        dim_x_layout = QVBoxLayout()
        dim_x_layout.addWidget(get_normal_label('Select dimension for x-axis.'))
        x_select = QHBoxLayout()
        x_select.addWidget(get_one_letter_normal_label(dim_x_str))
        self.combo_x_dim.addItems([display_names[NormSystem.dim_o], display_names[NormSystem.dim_r],
                              display_names[NormSystem.dim_s], display_names[NormSystem.dim_t]])
        self.combo_x_dim.currentTextChanged.connect(lambda state: self.update_axis(state, dim_x_str))
        x_select.addWidget(self.combo_x_dim)
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
                                   if v != display_names[NormSystem.hierarchy] and v != self.combo_x_dim.currentText()])
        self.combo_y_dim.currentTextChanged.connect(lambda state: self.update_axis(state, dim_y_str))
        y_select.addWidget(self.combo_y_dim)
        dim_y_layout.addLayout(y_select)

        self.checkbox_all.clicked.connect(self.update_show_both_two_dims)
        dim_y_layout.addWidget(self.checkbox_all)
        dim_settings_layout.addLayout(dim_y_layout, 1)

        # z- axis
        dim_z_layout = QVBoxLayout()
        dim_z_layout.addWidget(get_normal_label('Select dimension for z-axis.'))

        z_select = QHBoxLayout()
        z_select.addWidget(get_one_letter_normal_label(dim_z_str))

        self.combo_z_dim.addItems([v for v in display_names.values()
                                   if v != display_names[NormSystem.hierarchy] and v != self.combo_x_dim.currentText()])
        self.combo_z_dim.currentTextChanged.connect(lambda state: self.update_axis(state, dim_z_str))
        z_select.addWidget(self.combo_z_dim)
        self.combo_z_dim.setDisabled(True)
        dim_z_layout.addLayout(z_select)

        # Placeholder widget for vertical alignment
        dim_z_layout.addWidget(QWidget())
        dim_settings_layout.addLayout(dim_z_layout, 1)

        layout.addLayout(dim_settings_layout)

    def update_axis(self, value, dimension):
        """
        Reacts to setting an axis. Sets the internal value and, in case the x-axis was set, that dimension is removed
        for the choice of the y-axis.

        :param value: The new value
        :param dimension: the dimension
        """
        if value != '':  # happens if combobox is cleared
            key = list(filter(lambda x: display_names[x] == value, display_names))[0]
            if dimension == dim_x_str:
                self.x_axis = key
                y_text = self.combo_y_dim.currentText()
                z_text = self.combo_z_dim.currentText()
                if y_text != value: # keep y key as selected
                    if z_text != value and z_text!= none_selected_str: #keep z key as selected
                        y_texts_new = [v for v in display_names.values() if v != value and v!= z_text
                                           and v != display_names[NormSystem.hierarchy]]
                    else: # dont keep z key as selected
                        y_texts_new = [v for v in display_names.values() if v != value
                                       and v != display_names[NormSystem.hierarchy]]
                        z_text = none_selected_str
                else: # keep nothing
                    y_texts_new = [v for v in display_names.values() if v != value
                                   and v != display_names[NormSystem.hierarchy]]
                    y_text = none_selected_str
                    z_text = none_selected_str

                z_texts = [v for v in display_names.values() if v != value and
                           (v == none_selected_str or v != self.combo_y_dim.currentText())
                                   and v != display_names[NormSystem.hierarchy]]
                self.combo_y_dim.clear()
                self.combo_y_dim.addItems(y_texts_new)
                self.combo_y_dim.setCurrentText(y_text)
                self.combo_z_dim.clear()
                self.combo_z_dim.addItems(z_texts)
                self.combo_z_dim.setCurrentText(z_text)
                self.y_axis = list(filter(lambda x: display_names[x] == self.combo_y_dim.currentText(), display_names))[0]
                self.z_axis = list(filter(lambda x: display_names[x] == self.combo_z_dim.currentText(), display_names))[0]

            if dimension == dim_y_str:
                self.y_axis = key
                z_text = self.combo_z_dim.currentText()
                self.combo_z_dim.clear()
                z_items_new = [x for x in display_names.values() if
                               x != self.combo_x_dim.currentText() and
                               x != display_names[NormSystem.hierarchy] and
                               (x == none_selected_str or x != display_names[key])]
                if key == none_selected_str:
                    z_text = none_selected_str
                self.combo_z_dim.addItems(z_items_new)
                self.combo_z_dim.setCurrentText(z_text)
                self.z_axis = list(filter(lambda x: display_names[x] == self.combo_z_dim.currentText(), display_names))[0]
                if value == none_selected_str:
                    self.combo_z_dim.setDisabled(True)
                else:
                    self.combo_z_dim.setDisabled(False)
            if dimension == dim_z_str:
                self.z_axis = key

            self.parent.plot_current_setup()


    def update_show_both_two_dims(self, value):
        """
        Reacts to (de-)selecting the option to display all four dimensions for the two-dimensional case (2 figures)
        :param value: (bool) The set value
        """
        self.show_both_two_dims = value
        self.parent.plot_current_setup()

    def update_show_all_one_dim(self, value):
        """
        Reacts to (de-)selecting the option to show all dimensions while displaying one-dimensional figures
        (4 figures in total).

        :param value: (bool) the set value
        """
        self.show_all_one_dim = value
        self.combo_y_dim.setDisabled(value)
        self.checkbox_all.setDisabled(value)
        self.parent.plot_current_setup()

    def get_show_all_one_dim(self):
        return self.show_all_one_dim

    def get_show_both_two_dims(self):
        return self.show_both_two_dims

    def get_x_axis(self):
        return self.x_axis

    def get_y_axis(self):
        return self.y_axis

    def get_z_axis(self):
        return self.z_axis

    def set_norm_system(self, norm_system):
        self.norm_system = norm_system

class HighlightableButton(QPushButton):
    """
    Used with the hightlight group to show only one selected button at a time (changes colors)
    """
    def __init__(self, click_event, data, highlightgroup, pixmap, parent=None):
        QPushButton.__init__(self, parent)
        self.data = data
        self._whenClicked = click_event
        self.highlightgroup = highlightgroup
        self.highlightgroup.add_element(self)
        self.highlighted = False
        self.setStyleSheet("padding: 5px")
        pix_width = pixmap.width()
        pix_height = pixmap.height()
        scale_factor = pix_height / one_row_normal_label_height
        self.setIcon(QIcon(pixmap))
        self.setIconSize(QSize(int(pix_width*scale_factor), one_row_normal_label_height))

    def highlight(self, highlight):
        if highlight:
            self.setStyleSheet("background-color: lightgray; padding: 5px")
        else:
            self.setStyleSheet("padding: 5px")
        self.highlighted = highlight

    def mousePressEvent(self, event):
        self.highlightgroup.highlight_me(self, not self.highlighted)
        if self.highlighted:
            self._whenClicked(self.data)
        else:
            self._whenClicked(None)


class HighlightGroup:
    """
    Used with the HighlightableButton to only show one selected button in the group at a time.
    """
    def __init__(self):
        self.elements= []

    def clear(self):
        self.elements = []

    def add_element(self, element):
        self.elements.append(element)

    def highlight_me(self, element, value):
        if value:
            for el in self.elements:
                el.highlight(False)
        element.highlight(value)

class PixmapUpdater(QThread):
    """
    Used for updating the pixmap in background.
    """
    def __init__(self, pixmap, new_values):
        super().__init__()
        self.pixmap = pixmap
        self.new_values = (x for x in set(new_values) if x not in pixmap.keys())

    def run(self):
        for v in self.new_values:
            self.pixmap[v] = latex_to_qpixmap(v)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myappid = 'vino.gui.2'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    window = MainWindow()
    window.show()

    app.exec()
