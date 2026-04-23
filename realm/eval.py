import numpy as np
import torch
from queue import Queue
import json
import time
import datetime
import os
import random
import omnigibson as og
from omnigibson.macros import gm
from realm.environments.realm_environment_dynamic import RealmEnvironmentDynamic
from realm.inference import InferenceClient, extract_from_obs
from realm.logging import VideoRecorder, save_results_to_csv
import time

try:
    from PIL import Image
except ImportError:
    Image = None

import csv

def append_result_to_csv(result, csv_path):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

def _ensure_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is uint8, HWC, RGB so we can safely write JPG.
    Handles common cases: CHW->HWC, float->uint8, grayscale->RGB.
    """
    img = np.asarray(img)

    # CHW -> HWC (common torch format)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = img.transpose(1, 2, 0)

    # float -> uint8
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        # If looks like 0~1, scale to 0~255
        if img_f.size > 0 and img_f.max() <= 1.5:
            img_f = img_f * 255.0
        img = np.clip(img_f, 0, 255).astype(np.uint8)

    # grayscale -> RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def _save_jpg(path: str, img_u8: np.ndarray, quality: int = 95):
    """
    Save uint8 HWC image as JPG.
    Requires pillow in the environment: pip install pillow
    """
    if Image is None:
        raise RuntimeError("PIL not installed. Please `pip install pillow` inside the container.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_u8).save(path, format="JPEG", quality=quality, subsampling=0)

SUPPORTED_TASKS = [
    "put_green_block_in_bowl", #0
    "put_banana_into_box", #1
    "rotate_marker", #2
    "rotate_mug", #3
    "pick_spoon", #4
    "pick_water_bottle", #5
    "stack_cubes", #6
    "push_switch", #7
    "open_drawer", #8
    "close_drawer", #9
]

SUPPORTED_PERTURBATIONS = [
    'Default', #0
    'V-AUG', # 1
    'V-SC', # 2
    'V-VIEW', # 3
    'V-LIGHT', # 4
    'S-PROP', # 5
    'S-LANG', # 6
    'S-MO', # 7
    'S-AFF', # 8
    'S-INT', # 9
    'B-HOBJ', # 10
    'VB-POSE', # 11
    'VB-MOBJ', # 12
    'SB-NOUN', # 13
    'SB-VRB', # 14
    'VSB-NOBJ' # 15
]


def set_sim_config():
    gm.DEFAULT_SIM_STEP_FREQ = 15 # orignally 15
    gm.DEFAULT_RENDERING_FREQ = 15 # orignally 15
    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False # this needs to be off to avoid bug with sludge state during collision: https://github.com/StanfordVL/BEHAVIOR-1K/issues/1201
    gm.ENABLE_OBJECT_STATES = True # this needs to be on because push_switch task usees the ToggledOn state

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
        task_id=0,
        perturbation_id=0,
        repeats=2,
        max_steps=300,
        horizon=8,
        model="pi0_FAST",
        port=8000,
        log_dir="/app/logs",
        save_frames=1,
        save_video=1
):
    start = time.perf_counter()
    og.log.info(f"DEBUG: Begin eval: {time.perf_counter() - start:.4f}s")
    set_sim_config()

    # -------------------- Create the environment + client --------------------
    task = SUPPORTED_TASKS[task_id]
    perturbations = [SUPPORTED_PERTURBATIONS[perturbation_id]]

    os.makedirs(log_dir, exist_ok=True)
    if save_video:
        os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)
    if save_frames:
        os.makedirs(os.path.join(log_dir, "information"), exist_ok=True)

    model_type = model  # TODO: infer type from model name
    client = InferenceClient(model_type, port)
    og.log.info(f"DEBUG: Client connected: {time.perf_counter() - start:.4f}s")

    print("Now start realm environ dynamic")
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task=task,
        perturbations=perturbations
    )
    print("Now end realm environ dynamic")

    og.log.info(f"DEBUG: Env created: {time.perf_counter() - start:.4f}s")

    global_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results = []

    # -------------------- Information recording config --------------------
    # Record at ~20Hz wall-clock
    record_interval_sec = 0.05

    for run_id in range(repeats):
        # ------------------------ pre-configure each run --------------------------------
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        video_recorder = VideoRecorder(log_dir, timestamp, run_id) if save_video else None

        qpos = []
        actions = []
        action_buffer = Queue()

        obs, _ = env.reset()
        instruction = env.instruction
        client.reset()

        # -------------------- Create per-run information folder --------------------
        run_name = f"{timestamp}_{model}_rollout_{task}_{perturbations[0]}_{run_id}"

        if save_frames:
            info_dir = os.path.join(log_dir, "information", run_name)
            img_dir = os.path.join(info_dir, "image")
            img_sec_dir = os.path.join(info_dir, "image_sec")
            wrist_dir = os.path.join(info_dir, "wrist_image")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(img_sec_dir, exist_ok=True)
            os.makedirs(wrist_dir, exist_ok=True)

            with open(os.path.join(info_dir, "task.json"), "w", encoding="utf-8") as f:
                json.dump({"task": str(instruction)}, f, ensure_ascii=False)

            frames_jsonl_path = os.path.join(info_dir, "frames.jsonl")
            frames_f = open(frames_jsonl_path, "w", encoding="utf-8")
        else:
            frames_f = None

        last_record_time = -1e9
        k = 0  # recorded frame index

        # -------------------- Rollout loop --------------------
        obs, rew, terminated, truncated, info = env.warmup(obs)

        t = 0
        task_progression = 0.0
        task_progression_timestamps = []
        terminal_steps = 15

        while t < max_steps and terminal_steps > 0:
            base_im, base_im_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs)

            if action_buffer.empty():
                pred_action_chunk = client.infer(
                    instruction, base_im, base_im_second, wrist_im, robot_state, gripper_state,
                    use_base_im_second=(env.task_type == "open_close_drawer" if hasattr(env, "task_type") else False)
                )

                if len(pred_action_chunk.shape) == 2:
                    assert pred_action_chunk.shape[-1] == 8
                    for action in pred_action_chunk[:horizon]:
                        action = np.squeeze(action)
                        action_buffer.put(action)
                else:
                    action_buffer.put(pred_action_chunk)

            # Save frame to video (unchanged)
            if save_video:
                video_recorder.add_frame(base_im, wrist_im)

            # Build state
            state = np.concatenate([robot_state, [gripper_state]]).astype(np.float32)  # (8,)
            qpos.append(state)

            # Get action and build executed new_action
            raw_action = action_buffer.get()
            actions.append(raw_action)

            new_joint_action = raw_action.copy()[:7]
            new_gripper_state = 1 if raw_action[7] > 0.5 else -1  # Prediction: (1,0) -> Target: (1,-1)
            new_gripper_state = np.atleast_1d(np.array(new_gripper_state))
            new_action = np.concatenate((new_joint_action, new_gripper_state)).astype(np.float32)

            # -------------------- Record LeRobot-required fields (jpg/json) --------------------
            if save_frames:
                now = time.perf_counter()
                if (now - last_record_time) >= record_interval_sec:
                    last_record_time = now

                    base_u8 = _ensure_uint8_hwc(base_im)
                    base_sec_u8 = _ensure_uint8_hwc(base_im_second)
                    wrist_u8 = _ensure_uint8_hwc(wrist_im)

                    img_name = f"{k:06d}.jpg"
                    img_sec_name = f"{k:06d}.jpg"
                    wrist_name = f"{k:06d}.jpg"

                    _save_jpg(os.path.join(img_dir, img_name), base_u8)
                    _save_jpg(os.path.join(img_sec_dir, img_sec_name), base_sec_u8)
                    _save_jpg(os.path.join(wrist_dir, wrist_name), wrist_u8)

                    frame_obj = {
                        "index": k,
                        "image": f"image/{img_name}",
                        "image_sec": f"image_sec/{img_sec_name}",
                        "wrist_image": f"wrist_image/{wrist_name}",
                        "robot_state": robot_state.tolist(),
                        "gripper_state": gripper_state.tolist(),
                        "action": new_action.tolist(),
                    }
                    frames_f.write(json.dumps(frame_obj, ensure_ascii=False) + "\n")
                    k += 1

            # Step env
            obs, curr_task_progression, terminated, truncated, info = env.step(new_action)
            # og.log.info(f"{t}: {curr_task_progression}")

            if curr_task_progression > task_progression:
                task_progression = curr_task_progression
                task_progression_timestamps.append(t)
            if task_progression >= 1.0:
                terminal_steps -= 1
            t += 1

        # ------------------------------------------------------------------------------
        # results.append({
        #     "task": task,
        #     "perturbation": perturbations,
        #     "model": model,
        #     "real2sim": "Simulated",
        #     "task_progression": task_progression,
        #     "task_progression_timestamps": task_progression_timestamps,
        #     "binary_SR": 1.0 if task_progression == 1.0 else 0.0
        # })
        result = {
            "task": task,
            "perturbation": perturbations[0],
            "model": model,
            "instruction": instruction,
            "real2sim": "Simulated",
            "task_progression": task_progression,
            "task_progression_timestamps": str(task_progression_timestamps),
            "binary_SR": 1.0 if task_progression == 1.0 else 0.0
        }

        # results.append(result)

        # 🔥 新增：立即写入CSV
        report_dir = os.path.join(log_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)

        csv_path = os.path.join(
            report_dir,
            f"{model}_{task}_{perturbations[0]}_live.csv"
        )

        append_result_to_csv(result, csv_path)

        og.log.info(f"DEBUG: Run finished: {time.perf_counter() - start:.4f}s")

        if frames_f is not None:
            frames_f.close()

        if save_video:
            save_filename = os.path.join(log_dir, "videos", run_name)
            video_recorder.save_video(save_filename)
            video_recorder.cleanup()

    # ------------------------------------------------------------------------------
    # save_results_to_csv(results, log_dir + "/reports", global_timestamp, model, task, perturbations[0])
    og.log.info("Done!")
    og.log.info(f"DEBUG: CLEAN-UP done: {time.perf_counter() - start:.4f}s")
