from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.io_utils import load_json
from gsworld.constants import DTC_DIR

DTC_DATASET = dict()


def _load_dtc_dataset():
    global DTC_DATASET
    DTC_DATASET = {
        "model_data": load_json(DTC_DIR / f"DTC_objects_all_download_urls.json")["releases"],
    }


def get_dtc_builder(
    scene: ManiSkillScene, id: str, add_collision: bool = True, add_visual: bool = True
):
    if "DTC" not in DTC_DATASET:
        _load_dtc_dataset()
    model_db = DTC_DATASET["model_data"]["DTC"]["objects"]

    builder = scene.create_actor_builder()

    metadata = model_db[id]
    density = 10 # metadata.get("density", 1000)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]
    physical_material = None
    if add_collision:
        collision_file = str(DTC_DIR / f"collision_meshes/DTC_1_0_{id}_3d-asset_collision.ply") # TODO do not use glb, use processed file
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )
    if add_visual:
        visual_file = str(DTC_DIR / f"visual_glbs/DTC_1_0_{id}_3d-asset.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    return builder
