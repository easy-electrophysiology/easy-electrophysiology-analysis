from sys import platform
if platform == "darwin":
    from gui_macos.open_heka_dialog import Ui_open_heka_dialog
else:
    from gui_windows.open_heka_dialog import Ui_open_heka_dialog
from load_heka import load_heka
import neo
from PySide2.QtWidgets import QTreeWidgetItem, QDialog
from PySide2 import QtCore


class OpenHekaDialog(QDialog):
    """
    TreeWidget that emits signal of group and series index from a list of
    HEKA group / series when the series is double-clicked.
    """
    tree_clicked = QtCore.Signal(int, int)

    def __init__(self, mw, dict_of_groups_series):
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
        """
        Handle the user double-clicked to select an item to load
        """
        if item.childCount() != 0:  # ignore top level (i.e. Group)
            return

        group_idx = self.dia.tree_widget.indexFromItem(item.parent()).row()
        series_idx = self.dia.tree_widget.indexFromItem(item).row()

        self.mw.show_wait_cursor()
        self.tree_clicked.emit(group_idx, series_idx)


class OpenHeka:
    """
    Wrapper around custom Neo IO that wraps Load Heka module. The custom NeoIO will force the channel order to be
    according to the recording type (voltage clamp  or current clamp, e.g. if voltage clamp, Im first and Vm second).

    If two input channels are available, both will be loaded and the stimulation protocol will be ignored. Otherwise,
    if only one channel is available as well as the stimulus protocol, the stimulus protocol will be reconstructed
    and set as he second channel (name 'stimulation').
    """
    def __init__(self, mw, full_filename):

        self.full_filename = full_filename
        self.mw = mw
        self.group_idx = None
        self.series_idx = None

        try:
            with load_heka.LoadHeka(self.full_filename, only_load_header=True) as self.heka:
                dict_of_groups_series = self.heka.get_dict_of_group_and_series()

        except BaseException as e:
            self.mw.show_messagebox("Heka Load Error",
                                    e.__str__())
            return

        self.heka_dialog = OpenHekaDialog(self.mw, dict_of_groups_series)
        self.heka_dialog.tree_clicked.connect(self.save_group_series_idx)
        self.heka_dialog.exec()

    def get_reader_and_neo_block(self):

        if self.group_idx is None:
            return False, False

        reader = neo.HekaIO(self.full_filename, self.group_idx, self.series_idx)

        try:
            neo_block = reader.read_block(force_order_to_recording_mode=True)

        except BaseException as e:
            self.mw.show_messagebox("Heka Load Error",
                                    e.__str__())
            return False, False

        return reader, neo_block

    def save_group_series_idx(self, group_idx, series_idx):
        self.group_idx = group_idx
        self.series_idx = series_idx
        self.heka_dialog.close()
        self.heka_dialog = None
