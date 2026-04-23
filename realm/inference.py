import os
import sys

import numpy as np
import omnigibson as og
from openpi_client import websocket_client_policy, image_tools

try:
    from PIL import Image
except ImportError:
    Image = None


GR00T_MODEL_ALIASES = {
    "gr00t",
    "gr00t_n1.6",
    "gr00t_n1_6",
    "gr00t_n1.6_droid",
    "gr00t_n1_6_droid",
    "gr00t-n1.6-droid",
    "gr00t-n1_6-droid",
}


def _normalize_model_name(model_type: str) -> str:
    return model_type.strip().lower().replace("-", "_")


def _is_gr00t_model(model_type: str) -> bool:
    normalized = _normalize_model_name(model_type)
    return normalized in GR00T_MODEL_ALIASES or normalized.startswith("gr00t")


def _get_gr00t_policy_client():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    gr00t_root = os.path.join(repo_root, "Isaac-GR00T")
    if not os.path.isdir(gr00t_root):
        raise ImportError(
            f"Isaac-GR00T repo not found at {gr00t_root}. "
            "Clone it into the REALM workspace before using GR00T models."
        )
    if gr00t_root not in sys.path:
        sys.path.insert(0, gr00t_root)
    from gr00t.policy.server_client import PolicyClient

    return PolicyClient


def _resize_for_gr00t(image: np.ndarray) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required for GR00T inference. Install `pillow` first.")
    return np.asarray(Image.fromarray(image).resize((320, 180))).astype(np.uint8)


def extract_from_obs(obs: dict):
    base_im = obs['external']['external_sensor0']['rgb'].cpu().numpy()[..., :3]
    base_im_second = obs['external']['external_sensor1']['rgb'].cpu().numpy()[..., :3]
    wrist_im = obs['franka']['franka:gripper_link_camera:Camera:0']['rgb'].cpu().numpy()[..., :3]
    proprio = obs['franka']['proprio'].cpu().numpy()
    robot_state = proprio[:7]
    gripper_state = proprio[7] / 0.05  # 0 = open, 0.05 = closed
    return base_im, base_im_second, wrist_im, robot_state, gripper_state


class InferenceClient:
    def __init__(self, model_type, port, host="localhost"):
        self.model_type = model_type
        self.client = None
        self.is_gr00t = _is_gr00t_model(model_type)
        if model_type != "debug":
            og.log.info("Connecting to server...")
            if self.is_gr00t:
                policy_client_cls = _get_gr00t_policy_client()
                self.client = policy_client_cls(
                    host=host,
                    port=port,
                    strict=False,
                )
            else:
                self.client = websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
            og.log.info("Connected!")

    def reset(self):
        if self.client is None:
            return
        if self.is_gr00t:
            self.client.reset()
        else:
            self.client.reset()

    def infer(self, instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state, use_base_im_second=False):
        if self.model_type == "debug":
            pred_action_chunk = np.atleast_1d(np.zeros(8))
            return pred_action_chunk

        if self.is_gr00t:
            ext_image = base_im_second if use_base_im_second else base_im
            ext_image_resized = _resize_for_gr00t(ext_image)
            wrist_im_resized = _resize_for_gr00t(wrist_im)

            obs_dict = {
                "video.exterior_image_1_left": ext_image_resized[None, None, ...],
                "video.wrist_image_left": wrist_im_resized[None, None, ...],
                "state.joint_position": np.asarray(robot_state, dtype=np.float32)[None, None, ...],
                "state.gripper_position": np.atleast_1d(np.asarray(gripper_state, dtype=np.float32))[None, None, ...],
                "annotation.language.language_instruction": [instruction],
            }
            pred, _ = self.client.get_action(obs_dict)
            pred_action_chunk = np.concatenate(
                (
                    pred["action.joint_position"][0],
                    pred["action.gripper_position"][0],
                ),
                axis=-1,
            )
            return pred_action_chunk
        else:
            img_to_use = base_im_second if use_base_im_second else base_im

            obs_dict = {
                "prompt": instruction,
                "observation/joint_position": robot_state,
                "observation/gripper_position": np.atleast_1d(np.array(gripper_state)),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(img_to_use, 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(wrist_im, 224, 224)
            }
            pred = self.client.infer(obs_dict)
            pred_action_chunk = pred["actions"]
            return pred_action_chunk
