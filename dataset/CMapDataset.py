import os
import sys
import json
import math
import random
import trimesh
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


class CMapDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        robot_names: list = None,
        is_train: bool = True,
        debug_object_names: list = None,
        num_points: int = 512,
        object_pc_type: str = 'random',
        dataset_path: str = None,
        extra_metadata_path: str = None,
        clutter_dataset_path: str = None,
        clutter_mix: float = 0.0,
        depth_noise: float = 0.0,
    ):
        self.batch_size = batch_size
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']
        self.is_train = is_train
        self.num_points = num_points
        self.object_pc_type = object_pc_type
        self.clutter_mix = float(clutter_mix or 0.0)
        self.depth_noise = float(depth_noise or 0.0)
        self.clutter_samples = []
        if clutter_dataset_path:
            cpath = (
                clutter_dataset_path
                if os.path.isabs(clutter_dataset_path)
                else os.path.join(ROOT_DIR, clutter_dataset_path)
            )
            blob = torch.load(cpath, map_location="cpu", weights_only=False)
            self.clutter_samples = list(blob.get("samples", blob))

        self.hands = {}
        self.dofs = []
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(math.sqrt(self.hands[robot_name].dof))

        split_json_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/split_train_validate_objects.json')
        dataset_split = json.load(open(split_json_path))
        self.object_names = dataset_split['train'] if is_train else dataset_split['validate']
        if debug_object_names is not None:
            print("!!! Using debug objects !!!")
            self.object_names = debug_object_names

        default_dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset.pt')
        if dataset_path is not None:
            resolved = dataset_path if os.path.isabs(dataset_path) else os.path.join(ROOT_DIR, dataset_path)
        else:
            resolved = default_dataset_path
        metadata = list(torch.load(resolved, map_location='cpu')['metadata'])
        if extra_metadata_path is not None:
            extra_resolved = (
                extra_metadata_path
                if os.path.isabs(extra_metadata_path)
                else os.path.join(ROOT_DIR, extra_metadata_path)
            )
            if extra_resolved != resolved:
                extra_meta = torch.load(extra_resolved, map_location='cpu')['metadata']
                metadata.extend(m for m in extra_meta if m[2] == 'boyahand')
        self.metadata = [m for m in metadata if m[1] in self.object_names and m[2] in self.robot_names]
        if not self.is_train:
            self.combination = []
            for robot_name in self.robot_names:
                for object_name in self.object_names:
                    self.combination.append((robot_name, object_name))
            self.combination = sorted(self.combination)

        self.object_pcs = {}
        self.object_normals = {}
        if self.object_pc_type != 'fixed':
            for object_name in self.object_names:
                name = object_name.split('+')
                mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
                mesh = trimesh.load_mesh(mesh_path)
                object_pc, indices = mesh.sample(65536, return_index=True)
                pc_normals = mesh.face_normals[indices]
                self.object_pcs[object_name] = torch.tensor(object_pc, dtype=torch.float32)
                self.object_normals[object_name] = torch.tensor(pc_normals, dtype=torch.float32)
        else:
            print("!!! Using fixed object pcs !!!")
 
    def _matrix_to_pose_(self, rotation_matrix):
        """
        rotation_matrix is a list of [4, 4] tensor on cpu
        """
        num_link = len(rotation_matrix)
        poses = torch.zeros((num_link, 6), dtype=torch.float32)
        for link_id, T in enumerate(rotation_matrix):
            rot = T[:3, :3].numpy()
            trans = T[:3, 3].numpy()
            axis_angle = R.from_matrix(rot).as_rotvec()
            pose = np.concatenate((trans, axis_angle), axis=0)
            poses[link_id] = torch.from_numpy(pose)
        return poses

    def _label_scene(self, scene_pc, object_name, thresh=0.02):
        tgt = self.object_pcs[object_name]
        # subsample target for speed
        if tgt.shape[0] > 4096:
            tgt = tgt[torch.randperm(tgt.shape[0])[:4096]]
        d = torch.cdist(scene_pc, tgt).min(dim=1).values
        return (d > thresh).long()  # 0 target, 1 clutter

    def _maybe_noise(self, pc):
        if self.depth_noise <= 0:
            return pc
        return pc + torch.randn_like(pc) * self.depth_noise

    def __getitem__(self, index):
        """
        Train: sample a batch of data
        Validate: get (robot, object) from index, sample a batch of data
        """
        if self.is_train:

            robot_name_batch = []
            object_name_batch = []
            view_dir_batch = []
            robot_pc_initial_batch = []
            robot_pc_target_batch = []
            robot_links_pc_batch = []
            robot_link_se3_batch = []
            object_pc_batch = []
            object_normal_batch = []
            scene_pc_batch = []
            scene_label_batch = []
            initial_q_batch = []
            target_q_batch = []
            all_link_se3_target_batch = []
            all_link_se3_initial_batch = []
            all_link_vec_trans_target_batch = []
            all_link_vec_trans_initial_batch = []

            for idx in range(self.batch_size):

                use_clutter = (
                    self.clutter_mix > 0
                    and self.clutter_samples
                    and random.random() < self.clutter_mix
                )
                clutter_row = None
                if use_clutter:
                    clutter_row = random.choice(self.clutter_samples)
                    robot_name = clutter_row["robot"]
                    if robot_name not in self.robot_names:
                        use_clutter = False
                        clutter_row = None
                        robot_name = random.choice(self.robot_names)
                else:
                    robot_name = random.choice(self.robot_names)
                robot_name_batch.append(robot_name)
                hand = self.hands[robot_name]
                metadata_robot = [(m[0], m[1]) for m in self.metadata if m[2] == robot_name]

                if use_clutter:
                    target_q, object_name = clutter_row["q"], clutter_row["object"]
                else:
                    target_q, object_name = random.choice(metadata_robot)
                target_q_batch.append(target_q)
                object_name_batch.append(object_name)
                robot_links_pc_batch.append(hand.links_pc)

                if self.object_pc_type == 'fixed':
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                elif self.object_pc_type == 'random':
                    indices = torch.randperm(65536)[:self.num_points]
                    object_pc = self.object_pcs[object_name][indices]
                    object_normal = self.object_normals[object_name][indices]
                    object_pc += torch.randn(object_pc.shape) * 0.002
                else:  # 'partial', remove 50% points
                    sel = torch.randperm(65536)[: self.num_points * 2]
                    object_pc = self.object_pcs[object_name][sel]
                    object_normal = self.object_normals[object_name][sel]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    keep = torch.sort(proj).indices[self.num_points :]
                    object_pc = object_pc[keep]
                    object_normal = object_normal[keep]
                    view_dir_batch.append(direction)
                if self.object_pc_type != 'partial':
                    view_dir_batch.append(torch.zeros(3, dtype=torch.float32))

                if use_clutter:
                    scene_pc = clutter_row["scene_pc"].float()
                    if scene_pc.shape[0] != self.num_points:
                        idx = torch.randint(0, scene_pc.shape[0], (self.num_points,))
                        scene_pc = scene_pc[idx]
                    scene_label = self._label_scene(scene_pc, object_name)
                    tgt_idx = (scene_label == 0).nonzero(as_tuple=False).view(-1)
                    if tgt_idx.numel() >= 32:
                        pick = tgt_idx[torch.randint(0, tgt_idx.numel(), (self.num_points,))]
                        object_pc = scene_pc[pick]
                        object_normal = torch.zeros_like(object_pc)
                    view_dir_batch[-1] = clutter_row.get(
                        "view_dir", view_dir_batch[-1]
                    ).float()
                else:
                    scene_pc = object_pc.clone()
                    scene_label = torch.zeros(self.num_points, dtype=torch.long)

                object_pc = self._maybe_noise(object_pc)
                scene_pc = self._maybe_noise(scene_pc)
                object_pc_batch.append(object_pc)
                object_normal_batch.append(object_normal)
                scene_pc_batch.append(scene_pc)
                scene_label_batch.append(scene_label)

                robot_pc_target, all_link_se3_target = hand.get_transformed_links_pc(target_q)
                robot_pc_target_batch.append(robot_pc_target)
                all_link_se3_target_batch.append(all_link_se3_target)
                all_link_vec_trans_target_batch.append(
                    self._matrix_to_pose_(all_link_se3_target)
                )

                initial_q = hand.get_initial_q(target_q)
                initial_q_batch.append(initial_q)

                robot_pc_initial, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
                robot_pc_initial_batch.append(robot_pc_initial)
                all_link_se3_initial_batch.append(all_link_se3_initial)
                all_link_vec_trans_initial_batch.append(
                    self._matrix_to_pose_(all_link_se3_initial)
                )

            return {
                'robot_name': robot_name_batch,  # list(len = B): str
                'object_name': object_name_batch,  # list(len = B): str
                'robot_pc_initial': robot_pc_initial_batch,
                'robot_pc_target': robot_pc_target_batch,
                'robot_links_pc': robot_links_pc_batch,
                'object_pc': torch.stack(object_pc_batch),
                'object_pc_normal': torch.stack(object_normal_batch),
                'initial_se3': all_link_se3_initial_batch,
                'target_se3': all_link_se3_target_batch,
                'initial_vec': all_link_vec_trans_initial_batch,
                'target_vec': all_link_vec_trans_target_batch,
                'initial_q': initial_q_batch,
                'target_q': target_q_batch,
                'view_dir': torch.stack(view_dir_batch),
                'scene_pc': torch.stack(scene_pc_batch),
                'scene_label': torch.stack(scene_label_batch),
            }
            
        else:  # validate
            robot_name, object_name = self.combination[index]
            hand = self.hands[robot_name]
            initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
            object_pc_batch = torch.zeros([self.batch_size, self.num_points, 3], dtype=torch.float32)
            robot_links_pc_batch = []
            initial_se3_batch = []

            view_dir_batch = []
            scene_pc_batch = torch.zeros_like(object_pc_batch)
            scene_label_batch = torch.zeros(
                [self.batch_size, self.num_points], dtype=torch.long
            )

            for batch_idx in range(self.batch_size):
                initial_q = hand.get_initial_q()
                _, all_link_se3_initial = hand.get_transformed_links_pc(initial_q)
                initial_se3_batch.append(all_link_se3_initial)
                robot_links_pc_batch.append(hand.links_pc)

                if self.object_pc_type == 'partial':
                    indices = torch.randperm(65536)[:self.num_points * 2]
                    object_pc = self.object_pcs[object_name][indices]
                    direction = torch.randn(3)
                    direction = direction / torch.norm(direction)
                    proj = object_pc @ direction
                    _, indices = torch.sort(proj)
                    indices = indices[self.num_points:]
                    object_pc = object_pc[indices]
                    view_dir_batch.append(direction)
                else:
                    name = object_name.split('+')
                    object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
                    object_pc = torch.load(object_path)[:, :3]
                    view_dir_batch.append(torch.zeros(3, dtype=torch.float32))

                initial_q_batch[batch_idx] = initial_q
                object_pc_batch[batch_idx] = object_pc
                scene_pc_batch[batch_idx] = object_pc
                scene_label_batch[batch_idx] = 0

            B, N, DOF = self.batch_size, self.num_points, len(hand.pk_chain.get_joint_parameter_names())
            assert initial_q_batch.shape == (B, DOF), \
                f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
            assert object_pc_batch.shape == (B, N, 3), \
                f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"

            return {
                'robot_name': robot_name,  # str
                'object_name': object_name,  # str
                'initial_q': initial_q_batch,
                'initial_se3': torch.stack(initial_se3_batch, dim=0),
                'object_pc': object_pc_batch,
                'robot_links_pc': robot_links_pc_batch,
                'view_dir': torch.stack(view_dir_batch),
                'scene_pc': scene_pc_batch,
                'scene_label': scene_label_batch,
            }

    def __len__(self):
        if self.is_train:
            return math.ceil(len(self.metadata) / self.batch_size)
        else:
            return len(self.combination)


def custom_collate_fn(batch):
    return batch[0]


def create_dataloader(cfg, is_train):
    dataset = CMapDataset(
        batch_size=cfg.batch_size,
        robot_names=cfg.robot_names,
        is_train=is_train,
        debug_object_names=cfg.debug_object_names,
        object_pc_type=cfg.object_pc_type,
        dataset_path=getattr(cfg, 'dataset_path', None),
        extra_metadata_path=getattr(cfg, 'extra_metadata_path', None),
        clutter_dataset_path=getattr(cfg, 'clutter_dataset_path', None),
        clutter_mix=getattr(cfg, 'clutter_mix', 0.0),
        depth_noise=getattr(cfg, 'depth_noise', 0.0),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=custom_collate_fn,
        num_workers=cfg.num_workers,
        shuffle=is_train
    )
    return dataloader
