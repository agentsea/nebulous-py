from nebulous.processors.models import V1Scale, V1ScaleDown, V1ScaleUp, V1ScaleZero

DEFAULT_SCALE = V1Scale(
    up=V1ScaleUp(
        above_pressure=30,
        duration="1m",
    ),
    down=V1ScaleDown(
        below_pressure=2,
        duration="1m",
    ),
    zero=V1ScaleZero(
        duration="5m",
    ),
)

DEFAULT_MIN_REPLICAS = 1
DEFAULT_MAX_REPLICAS = 3
