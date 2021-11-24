import copy

from load_heka import load_heka
import neo
from gui_windows.open_heka_dialog import Ui_open_heka_dialog
from PySide2.QtWidgets import QTreeWidgetItem, QDialog
from PySide2 import QtCore

# TODO:@ change name to opener
class OpenHekaDialog(QDialog):

    tree_clicked = QtCore.Signal(int, int)

    def __init__(self, mw, dict_of_groups_series):  # TODO: SET MODAL
        super(OpenHekaDialog, self).__init__(parent=mw)

        self.mw = mw
        self.mw.show_standard_cursor()

        self.dia = Ui_open_heka_dialog()
        self.dia.setupUi(self)

        self.dia.tree_widget.setHeaderLabels(["Name"])

        items = []
        for group_idx, (group, series) in enumerate(dict_of_groups_series.items()):
            item = QTreeWidgetItem([group])

            for series_idx, s in enumerate(series):
                child = QTreeWidgetItem([s])
                item.addChild(child)
            items.append(item)

        self.dia.tree_widget.insertTopLevelItems(0, items)
        self.dia.tree_widget.itemDoubleClicked.connect(self.handle_item_double_clicked)

    def handle_item_double_clicked(self, item):

        if item.childCount() != 0:  # ignore top level (i.e. Group)
            return

        group_idx = self.dia.tree_widget.indexFromItem(item.parent()).row()
        series_idx = self.dia.tree_widget.indexFromItem(item).row()

        self.mw.show_wait_cursor()
        self.tree_clicked.emit(group_idx, series_idx)


class OpenHeka:  # TODO: doc somehwer that it will swap channel order
    """
    Wrapper around
    """
    def __init__(self, mw, full_filename):

        self.full_filename = full_filename
        self.mw = mw
        self.group_idx = None
        self.series_idx = None

        with load_heka.LoadHeka(self.full_filename, only_load_header=True) as self.heka:
            dict_of_groups_series = self.heka.get_dict_of_group_and_series()

        self.heka_dialog = OpenHekaDialog(self.mw, dict_of_groups_series)
        self.heka_dialog.tree_clicked.connect(self.save_group_series_idx)
        self.heka_dialog.exec()

    def get_reader_and_neo_block(self):

        if self.group_idx is None:
            return False, False

        reader = neo.HekaIO(self.full_filename, self.group_idx, self.series_idx)
        neo_block = reader.read_block(force_order_to_recording_mode=True)

        return reader, neo_block

    def save_group_series_idx(self, group_idx, series_idx):
        self.group_idx = group_idx
        self.series_idx = series_idx
        self.heka_dialog.close()
        self.heka_dialog = None
