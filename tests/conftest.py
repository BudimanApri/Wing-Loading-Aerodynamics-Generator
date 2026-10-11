import numpy as np
import pytest
from src.fluent_parser import Dataset
from src.geometry import WingGeometry
from src.force_integration import Reference, ForceConvention


@pytest.fixture
def reference():
    return Reference(1, 2, 0, 0)


@pytest.fixture
def convention():
    return ForceConvention(True, "body_to_fluid", "on_body", True, True, True)


@pytest.fixture
def geometry():
    return WingGeometry([-2, 0, 2], [[0,-2,0], [0,0,0], [0,2,0]],
                        [[1,-2,0], [1,0,0], [1,2,0]], [0,0,0])


@pytest.fixture
def uniform_wing():
    # Rectangular c=1, b=4, Δp=30 Pa => L=120 N, L'=30 N/m.
    nx, ny = 20, 40
    x, y = np.meshgrid((np.arange(nx)+.5)/nx, -2+(np.arange(ny)+.5)*4/ny)
    x, y = x.ravel(), y.ravel()
    n = len(x)
    area = (1/nx)*(4/ny)
    return Dataset({"x": np.tile(x,2), "y": np.tile(y,2), "z": np.repeat([.05,-.05],n),
        "p": np.repeat([-10.,20.],n), "cp": np.repeat([-5.,10.],n),
        "ax": np.zeros(n*2), "ay": np.zeros(n*2), "az": np.repeat([area,-area],n),
        "tx": np.ones(n*2), "ty": np.zeros(n*2), "tz": np.zeros(n*2)}, {})
