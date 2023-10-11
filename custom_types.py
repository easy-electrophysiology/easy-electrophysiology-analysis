from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6 import QtCore, QtWidgets

Info = List[Union[Dict, Literal[0], List]]

Direction = Union[Literal[1], Literal[-1]]

NpArray64 = NDArray[np.float64]

FalseOrFloat = Union[Literal[False], NpArray64]

Int = Union[int, np.integer]

Bool = Union[np.bool_, bool]

ImOrVm = Union[Literal["Im"], Literal["Vm"]]

FalseOrInt = Union[Literal[False], NDArray[np.integer]]

FalseOrDict = Union[Literal[False], Dict]

FalseOrIdxFloatFloat = Tuple[Union[Literal[False], Int], FalseOrFloat, FalseOrFloat]

SpinBox = Union[QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox]

Parent = Union[QtWidgets.QMainWindow, QtWidgets.QDialog]

Model = QtCore.QAbstractTableModel

TemplateNum = Literal["1", "2", "3"]

FalseOrPlot = Union[Literal[False], pg.PlotDataItem]
