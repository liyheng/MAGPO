from jumanji.registration import make, register
from mava.coordsum.env import CoordSum
"""Environment Registration"""


register(
    id=f"5x20-80-v0",
    entry_point="mava.coordsum.env:CoordSum",
    kwargs={
        "num_agents": 5,
        "num_actions": 20,
        "time_limit": 100,
        "maxval": 80,
    },
) 
register(
    id=f"3x30-50-v0",
    entry_point="mava.coordsum.env:CoordSum",
    kwargs={
        "num_agents": 3,
        "num_actions": 30,
        "time_limit": 100,
        "maxval": 50,
    },
)
register(
    id=f"3x10-30-v0",
    entry_point="mava.coordsum.env:CoordSum",
    kwargs={
        "num_agents": 3,
        "num_actions": 10,
        "time_limit": 100,
        "maxval": 30,
    },
)
register(
    id=f"8x15-100-v0",
    entry_point="mava.coordsum.env:CoordSum",
    kwargs={
        "num_agents": 8,
        "num_actions": 15,
        "time_limit": 100,
        "maxval": 100,
    },
)
