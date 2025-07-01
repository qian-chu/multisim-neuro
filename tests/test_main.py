import multisim
from multisim import Simulator
import pytest


def test_basic_import():
    assert hasattr(multisim, "__version__")