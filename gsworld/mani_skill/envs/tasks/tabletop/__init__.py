# xarm6
from .xarm6.align import AlignXArmEnv
from .xarm6.rotate_banana import BananaRotationXArmEnv
from .xarm6.spoon_on_board import SpoonOnBoardXArmEnv

# fr3
from .franka.align import AlignFr3Env
from .franka.pnp_box import PnpBoxFr3Env
from .franka.pour_mustard import PourMustardFr3Env
from .franka.stack import StackFr3Env