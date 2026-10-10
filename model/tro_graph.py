import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from omegaconf import OmegaConf
from theseus.geometry.so3 import SO3

from bps_torch.bps import bps_torch
from model.vqvae.vq_vae import VQVAE
from utils.rotation import *
from model.denoiser import GraphDenoiser, ResidualGraphDenoiser
from utils.hand_model import create_hand_model
from utils.kanatani_reliability import (
    identifiability_collision_loss,
    patch_plane_reliability,
)
from utils.link_occupancy import link_occupancy

class RobotGraph(nn.Module):

    def __init__(
        self,
        vqvae_cfg,
        vqvae_pretrain,
        object_patch,
        max_link_node,
        robot_links,
        inference_config,
        bps_config,
        N_t_training,
        diffusion_config,
        denoiser_config,
        embodiment,
        loss_config,
        mode="train",
        residual_config=None,
        occupancy_config=None,
    ):

        super(RobotGraph, self).__init__()
        # vqvae encoder
        self.vqvae = VQVAE(vqvae_cfg)
        if vqvae_pretrain is not None:
            state_dict = torch.load(vqvae_pretrain, map_location='cpu')
            self.vqvae.load_state_dict(state_dict)
            print(f"Loaded pretrained VQVAE from {vqvae_pretrain}.")
        # vqvae fixed
        for param in self.vqvae.parameters():
            param.requires_grad = False

        # meta
        self.embodiment = embodiment
        self.hand_dict = {}
        for hand_name in self.embodiment:
            self.hand_dict[hand_name] = create_hand_model(hand_name)

        self.object_patch = object_patch
        self.max_link_node = max_link_node
        
        # link embedding
        self.robot_links = robot_links
        self.link_embed_dim = bps_config.n_bps_points + 4   # link bps, centroid, scale
        self.bps = bps_torch(**bps_config)
        self.link_token_encoder = nn.Sequential(
            nn.Linear(self.link_embed_dim, self.link_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.link_embed_dim, self.link_embed_dim)
        )

        self.N_t_training = N_t_training
        self.init_diffusion(diffusion_config)
        self.denoiser = GraphDenoiser(
            M=diffusion_config["M"],
            object_patch=self.object_patch,
            max_link_node=self.max_link_node,
            **OmegaConf.to_container(denoiser_config, resolve=True)
        )
        rc = residual_config
        if rc is not None and not isinstance(rc, dict):
            rc = OmegaConf.to_container(rc, resolve=True)
        rc = rc or {}
        self.use_residual = bool(rc.get("enabled", False))
        self.p_drop_clutter = float(rc.get("drop_prob", 0.2))
        self.z_s_dim = int(rc.get("z_s_dim", 64))
        self.residual_denoiser = None
        if self.use_residual:
            rcfg = OmegaConf.to_container(denoiser_config, resolve=True)
            rcfg["num_layers"] = int(rc.get("num_layers", 2))
            self.residual_denoiser = ResidualGraphDenoiser(
                M=diffusion_config["M"],
                object_patch=self.object_patch,
                max_link_node=self.max_link_node,
                z_s_dim=self.z_s_dim,
                zero_init_out=bool(rc.get("zero_init_out", True)),
                **rcfg,
            )
            self.scene_token_mlp = nn.Sequential(
                nn.Linear(15, 64),
                nn.SiLU(),
                nn.Linear(64, self.z_s_dim),
            )
            self.obj_type_embed = nn.Embedding(2, 64)
            self.freeze_eps_base()
        oc = occupancy_config
        if oc is not None and not isinstance(oc, dict):
            oc = OmegaConf.to_container(oc, resolve=True)
        oc = oc or {}
        self.use_occupancy = bool(oc.get("enabled", False))
        self.occ_radius = float(oc.get("radius", 0.015))
        self.occ_weight = float(oc.get("weight", 1.0))
        if self.use_occupancy:
            # o^R sits in the same GraphDenoiser as SE3 (not a side MLP).
            pass
        self.mode = mode
        self.link_embeddings = self.construct_bps()
        if self.mode == "train":
            self.loss_config = loss_config
        elif self.mode == "test":
            inference_config = OmegaConf.to_container(inference_config, resolve=True)
            self.inference_mode = inference_config["inference_mode"]
            assert self.inference_mode in ["unconditioned", "palm_conditioned"]
            cg = inference_config.get("collision_guidance") or {}
            self.collision_guidance = bool(cg.get("enabled", False))
            self.cg_scale = float(cg.get("scale", 1.0))
            self.cg_margin = float(cg.get("contact_margin", 0.008))
            self.cg_knn = int(cg.get("knn", 32))
            if self.inference_mode == "palm_conditioned":
                self.palm_names = inference_config["palm_names"]
                self.interpolation_clip = inference_config["interpolation_clip"] * math.pi / 180
                self.interpolation_rate = inference_config["interpolation_rate"]

                self.rotation_error = inference_config["rotation_error"] * math.pi / 180
                self.t_star = self.get_start_timestamp()

    def construct_bps(self):
        
        link_embedding_dict = {}
        for embodiment, hand_model in self.hand_dict.items():
            links_pc = hand_model.links_pc
            embodiment_bps = []
            for link_name, link_pc in links_pc.items():
                centroid, scale = self._unit_ball_(link_pc)
                link_pc = (link_pc - centroid) / scale
                link_bps = self.bps.encode(
                    link_pc,
                    feature_type=['dists'],
                    x_features=None,
                    custom_basis=None
                )['dists']
                link_bps = torch.cat([link_bps, centroid, scale.view(1, 1)], dim=-1)
                embodiment_bps.append(link_bps)
            link_embedding_dict[embodiment] = torch.cat(
                embodiment_bps, dim=0
            )
        return link_embedding_dict

    def _encode_object(self, object_pc, knn=32):
        with torch.no_grad():
            normal_pc, centroids, scale = self._normalize_pc_(object_pc)
            object_tokens = self.vqvae.encode(normal_pc)
            ident_w, ident_n = patch_plane_reliability(
                normal_pc, object_tokens["xyz"], knn=knn
            )
        # Keep TRO's global scale channel. Do not replace it with ident.
        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, self.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )
        return object_nodes, normal_pc, object_tokens, ident_w, ident_n, scale, centroids
            
    def _unit_ball_(self, pc):

        centroid = torch.mean(pc, dim=0, keepdim=True)
        pc = pc - centroid
        max_radius = pc.norm(dim=-1).max()
        return centroid, max_radius

    def _normalize_pc_(self, pc):

        # recenter
        B, N, _ = pc.shape
        centroids = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroids

        scale, _ = torch.max(torch.abs(pc), dim=1, keepdim=True)
        scale, _ = torch.max(scale, dim=2, keepdim=True)
        pc = pc / scale

        return pc, centroids, scale

    def freeze_eps_base(self):
        self.denoiser.eval()
        for param in self.denoiser.parameters():
            param.requires_grad = False
        for param in self.vqvae.parameters():
            param.requires_grad = False
        for param in self.link_token_encoder.parameters():
            param.requires_grad = False

    def _pca_aw_mu(self, pc, eps=1e-8):
        mu = pc.mean(dim=1)
        x = pc - mu[:, None, :]
        n = max(int(pc.shape[1]) - 1, 1)
        cov = torch.matmul(x.transpose(1, 2), x) / n
        evals, evecs = torch.linalg.eigh(cov)
        w = evals.clamp_min(eps)
        a = evecs.reshape(pc.shape[0], 9)
        return a, w, mu

    def _scene_token(self, tgt_pc):
        a, w, mu = self._pca_aw_mu(tgt_pc)
        return self.scene_token_mlp(torch.cat([a, w, mu], dim=-1))

    def _resample_pc(self, pc, n, eps=1e-8):
        B, N, C = pc.shape
        if N == n:
            return pc
        if N == 0:
            return pc.new_zeros(B, n, 3)
        idx = torch.randint(0, N, (B, n), device=pc.device)
        return torch.gather(pc, 1, idx.unsqueeze(-1).expand(-1, -1, C))

    def _points_by_label(self, scene_pc, scene_label, label, n):
        B = scene_pc.shape[0]
        out = scene_pc.new_zeros(B, n, 3)
        has = scene_pc.new_zeros(B, dtype=torch.bool)
        for b in range(B):
            pts = scene_pc[b][scene_label[b] == label]
            if pts.shape[0] == 0:
                continue
            has[b] = True
            out[b] = self._resample_pc(pts.unsqueeze(0), n)[0]
        return out, has

    def _encode_object_in_frame(self, pc, centroids, scale, knn=32):
        pc_n = (pc - centroids) / scale.clamp_min(1e-8)
        with torch.no_grad():
            object_tokens = self.vqvae.encode(pc_n)
            ident_w, ident_n = patch_plane_reliability(
                pc_n, object_tokens["xyz"], knn=knn
            )
        object_nodes = torch.cat(
            [
                object_tokens["xyz"],
                scale.expand(-1, self.object_patch, -1),
                object_tokens["z_q"],
            ],
            dim=-1,
        )
        return object_nodes, pc_n, ident_w, ident_n

    def _apply_type_embed(self, nodes, patch_type):
        typed = nodes.clone()
        valid = patch_type >= 0
        safe = patch_type.clamp(min=0, max=1)
        emb = self.obj_type_embed(safe) * valid.unsqueeze(-1).to(nodes.dtype)
        typed[..., 4:] = typed[..., 4:] + emb
        return typed

    def _residual_object_graph(
        self, batch, object_pc, object_nodes, scale, centroids, drop_flag, knn=32
    ):
        device = object_pc.device
        B = object_pc.shape[0]
        scene_pc = batch.get("scene_pc", object_pc)
        scene_label = batch.get("scene_label")
        if scene_label is None:
            scene_label = torch.zeros(
                scene_pc.shape[0],
                scene_pc.shape[1],
                device=device,
                dtype=torch.long,
            )
        tgt_from_scene, has_tgt = self._points_by_label(
            scene_pc, scene_label, 0, object_pc.shape[1]
        )
        clt_pc, has_clt = self._points_by_label(
            scene_pc, scene_label, 1, object_pc.shape[1]
        )
        tgt_for_axis = torch.where(
            has_tgt[:, None, None], tgt_from_scene, object_pc
        )
        tgt_n = (tgt_for_axis - centroids) / scale.clamp_min(1e-8)
        z_s = self._scene_token(tgt_n)
        z_s = torch.where(drop_flag[:, None], torch.zeros_like(z_s), z_s)

        V_O_clt = object_nodes
        if bool(has_clt.any()):
            V_O_clt, _, _, _ = self._encode_object_in_frame(
                clt_pc, centroids, scale, knn=knn
            )
        keep_clt = has_clt & (~drop_flag)
        V_O_clt = V_O_clt * keep_clt[:, None, None].to(V_O_clt.dtype)
        type_tgt = torch.zeros(B, self.object_patch, device=device, dtype=torch.long)
        type_clt = torch.ones(B, self.object_patch, device=device, dtype=torch.long)
        type_clt = torch.where(
            keep_clt[:, None], type_clt, type_clt.new_full(type_clt.shape, -1)
        )
        if bool(keep_clt.any()):
            V_O_full = torch.cat([object_nodes, V_O_clt], dim=1)
            patch_type = torch.cat([type_tgt, type_clt], dim=1)
        else:
            V_O_full = object_nodes
            patch_type = type_tgt
        V_O_full = self._apply_type_embed(V_O_full, patch_type)
        return V_O_full, z_s, patch_type

    def _or_edges(self, noisy_V_R_se3, object_positions):
        B, P, _ = object_positions.shape
        dtype = object_positions.dtype
        device = object_positions.device
        object_se3 = (
            torch.eye(4, device=device, dtype=dtype)
            .expand(B, P, -1, -1)
            .clone()
        )
        object_se3[:, :, :3, 3] = object_positions
        return matrix_to_vector(
            compute_batch_relative_se3(noisy_V_R_se3, object_se3)
        )

    def _eps_base(self, V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t):
        with torch.no_grad():
            eps = self.denoiser(V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t)
        return eps.detach()

    def _eps_res(
        self,
        V_O_full,
        noisy_V_R,
        noisy_E_OR_full,
        noisy_E_RR,
        t,
        z_s,
        patch_type,
    ):
        return self.residual_denoiser(
            V_O_full,
            noisy_V_R,
            noisy_E_OR_full,
            noisy_E_RR,
            t,
            z_s=z_s,
            patch_type=patch_type,
        )

    def init_diffusion(self, cfg):

        self.M = cfg["M"]
        self.scheduling = cfg["scheduling"]
        if self.scheduling == "linear":
            self.beta_min, self.beta_max = cfg["beta_min"], cfg["beta_max"]
            betas = torch.linspace(self.beta_min, self.beta_max, self.M)
        else:
            raise NotImplementedError()
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer(
            "alpha_bars",
            torch.tensor([torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]),
        )
        self.ddim_steps = cfg["ddim_steps"]
        self.eta = cfg["ddim_eta"]
        self.noise_lambda = cfg["lambda"]

    def expand_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return (
            x[:, None, :, :]
            .expand(-1, self.N_t_training, -1, -1)
            .reshape(B * self.N_t_training, N, -1)
        )

    def _expand_and_reshape_(self, x, name):

        shape = x.shape
        B = x.shape[0]
        if len(shape) == 3:  # Node
            return (
                x[:, None, :, :]
                .expand(-1, self.N_t_training, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2])
            )
        elif len(shape) == 4:  # Edge
            return (
                x[:, None, :, :, :]
                .expand(-1, self.N_t_training, -1, -1, -1)
                .reshape(B * self.N_t_training, shape[1], shape[2], shape[3])
            )
        elif len(shape) == 2:  # Node Mask
            return (
                x[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(B * self.N_t_training, shape[1])
            )
        else:
            raise ValueError(f"Unsupported shape for {name}: {shape}")

    def forward(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        ## Graph Construction

        ident_knn = int(self.loss_config.get("ident_knn", 32))
        object_nodes, normal_pc, object_tokens, ident_w_node, ident_n_node, scale, centroids = self._encode_object(
            object_pc, knn=ident_knn
        )

        # Target Link Node
        target_vec = batch["target_vec"]
        link_target_poses = torch.zeros(
            [B, self.max_link_node, 6], device=device, dtype=dtype
        )
        link_robot_embeds = torch.zeros(
            [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
        )
        link_node_masks = torch.zeros(
            [B, self.max_link_node], device=device, dtype=torch.bool
        )
        
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            target_pose_vec = target_vec[b]
            target_trans = target_pose_vec[:, :3]
            target_rot = target_pose_vec[:, 3:]

            object_center, object_scale = centroids[b], scale[b]
            target_trans = (target_trans - object_center) / object_scale
            target_pose_vec = torch.cat([target_trans, target_rot], dim=-1)
            link_target_poses[b, :num_link, :] = target_pose_vec

            robot_name = batch["robot_name"][b]
            link_bps = self.link_embeddings[robot_name]
            link_embed = self.link_token_encoder(link_bps)
            link_robot_embeds[b, :num_link, :] = link_embed
            link_node_masks[b, :num_link] = True

        robot_nodes = torch.cat([
            link_target_poses, 
            link_robot_embeds
        ], dim=-1)  # [B, L, 6+64]

        # Robot-Robot Edge, Fixed Pose
        link_rel_poses = torch.zeros(
            [B, self.max_link_node, self.max_link_node, 6],
            device=device, dtype=dtype
        )
        link_rel_masks = torch.zeros(
            [B, self.max_link_node, self.max_link_node],
            device=device, dtype=torch.bool
        )
        norm_robot_vec = robot_nodes[:, :, :6]
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T = vector_to_matrix(norm_robot_vec[b])[:num_link]
            rel_pose = compute_relative_se3(T, T)
            link_rel_poses[b, :num_link, :num_link, :] = matrix_to_vector(rel_pose)
            link_rel_masks[b, :num_link, :num_link] = True
            link_rel_masks[b, torch.arange(num_link), torch.arange(num_link)] = False

        # Robot-Object Edge
        link_object_rel_poses = torch.zeros(
            [B, self.max_link_node, self.object_patch, 6],
            device=device, dtype=dtype
        )
        link_object_masks = torch.zeros(
            [B, self.max_link_node, self.object_patch],
            device=device, dtype=torch.bool
        )
        # Only consider object translation
        object_positions = object_nodes[:, :, :3]
        object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, self.object_patch, -1, -1).clone()
        object_se3[:, :, :3, 3] = object_positions   # [B, P, 4, 4]
        for b in range(B):
            robot_name = batch["robot_name"][b]
            num_link = self.robot_links[robot_name]
            T_r = vector_to_matrix(norm_robot_vec[b])[:num_link]   # [L, 4, 4]
            T_o = object_se3[b]                                    # [P, 4, 4]
            T_rel = compute_relative_se3(T_r, T_o)
            T_rel = matrix_to_vector(T_rel)
            link_object_rel_poses[b, :num_link, :self.object_patch] = T_rel
            link_object_masks[b, :num_link, :self.object_patch] = True

        ## Forward Diffusion
        t = np.random.randint(0, self.M, (B * self.N_t_training))
        V_O = self._expand_and_reshape_(object_nodes, "V_O")             # [B*T, P, 68]
        V_R = self._expand_and_reshape_(robot_nodes, "V_R")              # [B*T, L, 70]
        V_R_trans, V_R_rot, V_R_embed = V_R[:, :, :3], V_R[:, :, 3:6], V_R[:, :, 6:]  

        eta_V_R_trans = torch.randn_like(V_R_trans)
        eta_V_R_rot = torch.randn_like(V_R_rot)
        a_bar = self.alpha_bars[t][:, None, None]
        
        noisy_trans = a_bar.sqrt() * V_R_trans + (1 - a_bar).sqrt() * eta_V_R_trans
        noisy_rot = a_bar.sqrt() * V_R_rot + (1 - a_bar).sqrt() * eta_V_R_rot
        embed64 = V_R_embed[..., :64]
        eta_o = None
        if self.use_occupancy and "scene_pc" in batch:
            scene_pc = batch["scene_pc"]
            scene_label = batch["scene_label"]
            link_xyz = robot_nodes[:, :, :3] * scale + centroids
            o_clean = link_occupancy(link_xyz, scene_pc, scene_label, self.occ_radius)
            o_t = self._expand_and_reshape_(o_clean, "o")
            eta_o = torch.randn_like(o_t)
            noisy_o = a_bar.sqrt() * o_t + (1 - a_bar).sqrt() * eta_o
            noisy_V_R = torch.cat([noisy_trans, noisy_rot, noisy_o, embed64], dim=-1)
        else:
            noisy_V_R = torch.cat([noisy_trans, noisy_rot, V_R_embed], dim=-1)

        # update graph edges
        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])           # [B*T, L, 4, 4] 
        noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
            compute_batch_relative_se3(                       
                noisy_V_R_se3, noisy_V_R_se3
            )
        )
        object_positions = V_O[:, :, :3]
        B, P, _ = object_positions.shape
        object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
        object_se3[:, :, :3, 3] = object_positions                                                   
        noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
        noisy_E_OR = matrix_to_vector(noisy_E_OR)

        ## Backward Denoising
        n_batch = object_pc.shape[0]
        drop_t = None
        eps_res = noisy_V_R.new_zeros(noisy_V_R.shape[0], noisy_V_R.shape[1], 6)
        if self.use_residual:
            ident_knn = int(self.loss_config.get("ident_knn", 32))
            drop_flag = torch.rand(n_batch, device=device) < self.p_drop_clutter
            if not self.training:
                drop_flag = drop_flag.new_zeros(n_batch, dtype=torch.bool)
            V_O_full, z_s, patch_type = self._residual_object_graph(
                batch, object_pc, object_nodes, scale, centroids, drop_flag, knn=ident_knn
            )
            V_O_full = self._expand_and_reshape_(V_O_full, "V_O_full")
            patch_type_t = (
                patch_type[:, None, :]
                .expand(-1, self.N_t_training, -1)
                .reshape(n_batch * self.N_t_training, patch_type.shape[1])
            )
            z_s_t = z_s.repeat_interleave(self.N_t_training, dim=0)
            drop_t = drop_flag.repeat_interleave(self.N_t_training, dim=0)
            noisy_E_OR_full = self._or_edges(noisy_V_R_se3, V_O_full[:, :, :3])

            eps_base = self._eps_base(V_O, noisy_V_R, noisy_E_OR, noisy_E_RR, t)
            eps_res = self._eps_res(
                V_O_full,
                noisy_V_R,
                noisy_E_OR_full,
                noisy_E_RR,
                t,
                z_s_t,
                patch_type_t,
            )
            pred_link_noise = eps_base + eps_res
        else:
            pred_link_noise = self.denoiser(
                V_O,
                noisy_V_R,
                noisy_E_OR,
                noisy_E_RR,
                t
            )
            drop_t = None
            eps_base = pred_link_noise

        # noise loss
        M_V_R = self._expand_and_reshape_(link_node_masks, "M_V_R").float()
        pred_trans_noise = pred_link_noise[:, :, :3]
        pred_rot_noise = pred_link_noise[:, :, 3:6]
        pred_eta_o = pred_link_noise[:, :, 6:9] if pred_link_noise.shape[-1] >= 9 else None

        error_trans_noise = (eta_V_R_trans - pred_trans_noise) ** 2
        error_trans_noise = error_trans_noise.mean(dim=-1)
        loss_trans_noise = (error_trans_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        error_rot_noise = (eta_V_R_rot - pred_rot_noise) ** 2
        error_rot_noise = error_rot_noise.mean(dim=-1)
        loss_rot_noise = (error_rot_noise * M_V_R).sum() / (M_V_R.sum() + eps)

        loss_res0 = loss_trans_noise.new_tensor(0.0)
        eps_res_norm = loss_trans_noise.new_tensor(0.0)
        if self.use_residual:
            eps_res_norm = (eps_res.pow(2).mean(dim=-1) * M_V_R).sum() / (
                M_V_R.sum() + eps
            )
            if drop_t is not None and bool(drop_t.any()):
                w_drop = M_V_R * drop_t[:, None].to(M_V_R.dtype)
                loss_res0 = (eps_res.pow(2).mean(dim=-1) * w_drop).sum() / (
                    w_drop.sum() + eps
                )

        loss_occ = loss_trans_noise.new_tensor(0.0)
        if pred_eta_o is not None and eta_o is not None:
            error_o = (eta_o - pred_eta_o) ** 2
            loss_occ = (error_o.mean(dim=-1) * M_V_R).sum() / (M_V_R.sum() + eps)

        ident_weight = float(self.loss_config.get("ident_weight", 0.0))
        loss_ident = loss_trans_noise.new_tensor(0.0)
        if ident_weight > 0:
            patch_xyz = V_O[:, :, :3]
            ident_knn = int(self.loss_config.get("ident_knn", 32))
            contact_margin = float(self.loss_config.get("ident_contact_margin", 0.008))
            ident_t_max = int(self.loss_config.get("ident_t_max", 200))
            t_dev = torch.as_tensor(t, device=device)
            low_t = t_dev < ident_t_max
            if low_t.any():
                ident_w = ident_w_node.repeat_interleave(self.N_t_training, dim=0)
                ident_n = ident_n_node.repeat_interleave(self.N_t_training, dim=0)

                pred_x0_trans = (
                    noisy_trans - (1 - a_bar).sqrt() * pred_trans_noise
                ) / a_bar.sqrt().clamp_min(eps)

                ident_acc = loss_trans_noise.new_tensor(0.0)
                n_used = 0
                for b in torch.where(low_t)[0].tolist():
                    orig_b = b // self.N_t_training
                    robot_name = batch["robot_name"][orig_b]
                    n_link = self.robot_links[robot_name]
                    ident_acc = ident_acc + identifiability_collision_loss(
                        pred_x0_trans[b : b + 1, :n_link],
                        patch_xyz[b : b + 1],
                        ident_w[b : b + 1],
                        ident_n[b : b + 1],
                        contact_margin=contact_margin,
                    )
                    n_used += 1
                loss_ident = ident_acc / max(n_used, 1)

        vis_weight = float(self.loss_config.get("vis_weight", 0.0))
        loss_vis = loss_trans_noise.new_tensor(0.0)
        view_dir = batch.get("view_dir")
        if vis_weight > 0 and view_dir is not None:
            vd = view_dir
            if vd.dim() == 2 and vd.abs().sum() > 0:
                pred_x0_trans = (
                    noisy_trans - (1 - a_bar).sqrt() * pred_trans_noise
                ) / a_bar.sqrt().clamp_min(eps)
                vd_t = vd.repeat_interleave(self.N_t_training, dim=0)
                vis_acc = loss_trans_noise.new_tensor(0.0)
                n_used = 0
                for b in range(pred_x0_trans.shape[0]):
                    orig_b = b // self.N_t_training
                    if vd[orig_b].abs().sum() < 1e-8:
                        continue
                    robot_name = batch["robot_name"][orig_b]
                    n_link = self.robot_links[robot_name]
                    center = pred_x0_trans.new_zeros(1, 3)
                    vis_acc = vis_acc + visibility_approach_loss(
                        pred_x0_trans[b : b + 1, :n_link],
                        vd_t[b : b + 1],
                        center,
                    )
                    n_used += 1
                if n_used:
                    loss_vis = vis_acc / n_used

        occ_w = float(getattr(self, "occ_weight", 0.0)) if getattr(self, "use_occupancy", False) else 0.0
        total_loss = (
            self.loss_config["trans_weight"] * loss_trans_noise
            + self.loss_config["rot_weight"] * loss_rot_noise
            + ident_weight * loss_ident
            + vis_weight * loss_vis
            + loss_res0
            + occ_w * loss_occ
        )
        loss_dict = {
            "loss_rot": loss_rot_noise,
            "loss_trans": loss_trans_noise,
            "loss_ident": loss_ident,
            "loss_vis": loss_vis,
            "loss_res0": loss_res0,
            "loss_occ": loss_occ,
            "eps_res_norm": eps_res_norm,
            "loss_total": total_loss,
        }
        return loss_dict


    def _predict_noise_infer(
        self,
        batch,
        object_pc,
        object_nodes,
        scale,
        centroids,
        noisy_V_R,
        noisy_E_OR,
        noisy_E_RR,
        t,
    ):
        eps = self.denoiser(
            object_nodes, noisy_V_R, noisy_E_OR, noisy_E_RR, t
        )
        if not self.use_residual:
            return eps
        drop_flag = object_pc.new_zeros(object_pc.shape[0], dtype=torch.bool)
        knn = int(getattr(self, "cg_knn", 32))
        V_O_full, z_s, patch_type = self._residual_object_graph(
            batch, object_pc, object_nodes, scale, centroids, drop_flag, knn=knn
        )
        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])
        noisy_E_OR_full = self._or_edges(noisy_V_R_se3, V_O_full[:, :, :3])
        return eps + self._eps_res(
            V_O_full, noisy_V_R, noisy_E_OR_full, noisy_E_RR, t, z_s, patch_type
        )

    def get_start_timestamp(self, mu=1.596, eps=1e-8):
    
        target = torch.tensor(self.rotation_error / mu).clamp_min(eps) ** 2
        idx = torch.argmin(torch.abs((1.0 - self.alpha_bars) - target)).item()
        return int(idx)

    @torch.no_grad()
    def drop_half_halfspace(
        points: torch.Tensor,
        generator: torch.Generator | None = None,
        eps=1e-8
    ):
      
        B, N, _ = points.shape
        device = points.device
        k_remove = N // 2

        normals = torch.randn(B, 3, device=device, generator=generator)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + eps)

        proj = (points * normals[:, None, :]).sum(-1)
        idx_remove = proj.topk(k_remove, dim=1, largest=True, sorted=False).indices
        keep_mask = torch.ones((B, N), dtype=torch.bool, device=device)
        keep_mask[torch.arange(B, device=device)[:, None], idx_remove] = False

        new_points = points[keep_mask].view(B, N - k_remove, 3)
        return new_points

    def inference(self, batch, eps=1e-8):

        object_pc = batch["object_pc"]
        B = object_pc.shape[0]
        device = object_pc.device
        dtype = object_pc.dtype

        robot_name = batch["robot_name"]
        link_names = list(batch["robot_links_pc"][0].keys())
        valid_links = self.robot_links[robot_name]

        # Object Node: xyz + Kanatani plane-ident + VQ code
        object_nodes, normal_pc, object_tokens, ident_w, ident_n, scale, centroids = self._encode_object(
            object_pc, knn=getattr(self, "cg_knn", 32)
        )
        ident_w = ident_w.detach()
        ident_n = ident_n.detach()
      
        if self.inference_mode == "unconditioned":

            ## Standard DDIM inference
            step = self.M // self.ddim_steps
            ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
            all_diffuse_step_poses_dict = {}

            ## Start from complete noise of link node (pose)
            noisy_V_R_trans = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            noisy_V_R_rot = torch.randn(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            link_robot_embeds = torch.zeros(
                [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
            )
            for b in range(B):
                robot_name = batch["robot_name"]
                num_link = self.robot_links[robot_name]
                link_bps = self.link_embeddings[robot_name]
                link_embed = self.link_token_encoder(link_bps)
                link_robot_embeds[b, :num_link, :] = link_embed
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)

            ## Formulate edges
            noisy_V_R_pose = noisy_V_R[:, :, :6]
            noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
            noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                compute_batch_relative_se3(                       
                    noisy_V_R_se3, noisy_V_R_se3
                )
            )
            object_positions = object_nodes[:, :, :3]
            B, P, _ = object_positions.shape
            object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
            object_se3[:, :, :3, 3] = object_positions                                                   
            noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
            noisy_E_OR = matrix_to_vector(noisy_E_OR)

            for i, diffuse_step in enumerate(ddim_t):
                
                diffuse_step = diffuse_step.item()
                # predict noise
                pred_link_pose_noise = self._predict_noise_infer(
                    batch,
                    object_pc,
                    object_nodes,
                    scale,
                    centroids,
                    noisy_V_R,
                    noisy_E_OR,
                    noisy_E_RR,
                    t=torch.full(
                        (object_nodes.shape[0],),
                        diffuse_step,
                        dtype=torch.long,
                        device=object_nodes.device
                    )
                )
                pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
                pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]
                
                # predict x_0
                a_bar_t = self.alpha_bars[diffuse_step]
                if i == len(ddim_t) - 1:
                    a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
                else:
                    a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

                x_t_trans = noisy_V_R_trans
                x_t_rot = noisy_V_R_rot
                x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise) / a_bar_t.sqrt()
                x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()

                if getattr(self, "collision_guidance", False):
                    n_link = self.robot_links[robot_name]
                    with torch.enable_grad():
                        x_guided = x_0_trans.detach().requires_grad_(True)
                        guide_loss = identifiability_collision_loss(
                            x_guided[:, :n_link],
                            object_nodes[:, :, :3],
                            ident_w,
                            ident_n,
                            contact_margin=self.cg_margin,
                        )
                        grad = torch.autograd.grad(guide_loss, x_guided)[0]
                    step = self.cg_scale * (1.0 - a_bar_t).clamp_min(0.05)
                    x_0_trans = x_0_trans - step * grad.detach()
                    pred_link_trans_noise = (
                        x_t_trans - a_bar_t.sqrt() * x_0_trans
                    ) / (1 - a_bar_t).sqrt().clamp_min(1e-8)

                sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
                ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

                if i == len(ddim_t) - 1:
                    z_trans = torch.zeros_like(x_0_trans)
                    z_rot = torch.zeros_like(x_0_rot)
                else:
                    z_trans = torch.randn_like(x_0_trans)
                    z_rot = torch.randn_like(x_0_rot)

                x_prev_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeffient * pred_link_trans_noise + sigma_t * z_trans * self.noise_lambda
                x_prev_rot = a_bar_prev.sqrt() * x_0_rot + ddim_coeffient * pred_link_rot_noise + sigma_t * z_rot * self.noise_lambda

                noisy_V_R_trans = x_prev_trans
                noisy_V_R_rot = x_prev_rot

                # update node and edge
                noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)
                noisy_V_R_pose = noisy_V_R[:, :, :6]
                noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
                noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                    compute_batch_relative_se3(                       
                        noisy_V_R_se3, noisy_V_R_se3
                    )
                )
                object_positions = object_nodes[:, :, :3]
                B, P, _ = object_positions.shape
                object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
                object_se3[:, :, :3, 3] = object_positions                                                   
                noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
                noisy_E_OR = matrix_to_vector(noisy_E_OR)

                # save snapshot
                pred_trans = noisy_V_R_trans * scale + centroids
                pred_rot = noisy_V_R_rot
                pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

                predict_link_pose_dict = {}
                denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
                predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])       # [B, L, 4, 4]
                for link_id, link_name in enumerate(link_names):
                    predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
                all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict
        
        elif self.inference_mode == "palm_conditioned":
            palm_name = self.palm_names[robot_name]
            palm_index = link_names.index(palm_name)
            palm_r3 = batch["initial_se3"][:, palm_index][:, :3, :3]
            initial_pose = matrix_to_vector(batch["initial_se3"])

            step = self.M // self.ddim_steps
            ddim_t = torch.arange(self.M - 1, -1, -step, device=device, dtype=torch.long)
            start_idx = int(torch.argmin(torch.abs(ddim_t - self.t_star)).item())
            t_start = int(ddim_t[start_idx].item())
            a_bar_s = self.alpha_bars[t_start]
            
            # initial links
            link_robot_rots = torch.zeros(
                [B, self.max_link_node, 3], device=device, dtype=dtype
            )
            link_robot_embeds = torch.zeros(
                [B, self.max_link_node, self.link_embed_dim], device=device, dtype=dtype
            )
            for b in range(B):
                robot_name = batch["robot_name"]
                num_link = self.robot_links[robot_name]
                link_bps = self.link_embeddings[robot_name]
                link_embed = self.link_token_encoder(link_bps)
                link_robot_embeds[b, :num_link, :] = link_embed
                link_robot_rots[b, :num_link] = initial_pose[b, :, 3:]

            noisy_V_R_rot = a_bar_s.sqrt() * link_robot_rots + (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
            noisy_V_R_trans = (1.0 - a_bar_s).sqrt() * torch.randn_like(link_robot_rots)
            noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)

            ## Formulate edges
            noisy_V_R_pose = noisy_V_R[:, :, :6]
            noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
            noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                compute_batch_relative_se3(                       
                    noisy_V_R_se3, noisy_V_R_se3
                )
            )
            object_positions = object_nodes[:, :, :3]
            B, P, _ = object_positions.shape
            object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
            object_se3[:, :, :3, 3] = object_positions                                                   
            noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
            noisy_E_OR = matrix_to_vector(noisy_E_OR)

            all_diffuse_step_poses_dict = {}
            for i in range(start_idx, len(ddim_t)):
                
                diffuse_step = int(ddim_t[i].item())
                # predict noise
                pred_link_pose_noise = self._predict_noise_infer(
                    batch,
                    object_pc,
                    object_nodes,
                    scale,
                    centroids,
                    noisy_V_R,
                    noisy_E_OR,
                    noisy_E_RR,
                    t=torch.full(
                        (object_nodes.shape[0],),
                        diffuse_step,
                        dtype=torch.long,
                        device=object_nodes.device
                    )
                )
                pred_link_trans_noise = pred_link_pose_noise[:, :, :3]
                pred_link_rot_noise = pred_link_pose_noise[:, :, 3:]
                
                # predict x_0
                a_bar_t = self.alpha_bars[diffuse_step]
                x_t_trans = noisy_V_R_trans
                x_t_rot = noisy_V_R_rot
                x_0_trans = (x_t_trans - (1 - a_bar_t).sqrt() * pred_link_trans_noise) / a_bar_t.sqrt()
                x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()

                #########Add Palm Rotation Guidance#############
                progress = (i + 1) / self.ddim_steps
                s_t = self.interpolation_rate * math.sin(0.5 * progress * math.pi)                                                  # timestamp ratio
          
                with torch.enable_grad():
                    pred_link_rot_noise_with_grad = pred_link_rot_noise.detach().clone().requires_grad_(True)
                    x_0_rot_with_grad = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise_with_grad) / a_bar_t.sqrt()

                    B, L, _ = x_0_rot_with_grad.shape
                    R_cur_all = SO3.exp_map(x_0_rot_with_grad.reshape(-1, 3)).to_matrix().reshape(B, L, 3, 3)

                    r_cur = x_0_rot_with_grad[:, palm_index]
                    R_cur = SO3.exp_map(r_cur).to_matrix()                 
                    R_init = palm_r3.detach().expand(B, 3, 3)
    
                    ###### Interpolation ############
                    R_err = R_init @ torch.linalg.inv(R_cur)
                    r_err = SO3(tensor=R_err).log_map()

                    theta = r_err.norm(dim=-1, keepdim=True).clamp_min(eps)
                    step_angle = torch.clamp(theta, max=self.interpolation_clip)
                    # delta_rate = s_t * (step_angle / theta)
                    delta_rate = s_t
                    
                    r_delta = delta_rate * r_err
                    R_delta = SO3.exp_map(r_delta).to_matrix()[:, None, :, :]
                    R_intered = torch.matmul(R_delta, R_cur_all)
                    r_intered = SO3(tensor=R_intered.reshape(-1, 3, 3)).log_map().reshape(B, L, 3)
                    pred_link_rot_noise = (x_t_rot - a_bar_t.sqrt() * r_intered) / (1 - a_bar_t).sqrt()

                    palm_rot_loss = rotation_matrix_geodesic_loss(R_cur, palm_r3)
                    x_0_rot = (x_t_rot - (1 - a_bar_t).sqrt() * pred_link_rot_noise) / a_bar_t.sqrt()
      
                    # print(f"DDIM {diffuse_step} | Palm Loss: {palm_rot_loss.item():.4f} | Delta Rate: {delta_rate}")
                ################################################

                if i == len(ddim_t) - 1:
                    a_bar_prev = torch.tensor(1.0, device=device, dtype=dtype)
                else:
                    a_bar_prev = self.alpha_bars[ddim_t[i + 1]]

                sigma_t = self.eta * torch.sqrt(((1 - a_bar_prev) / (1 - a_bar_t)) * (1 - a_bar_t / a_bar_prev))            
                ddim_coeffient = torch.sqrt(1 - a_bar_prev - sigma_t ** 2)

                if i == len(ddim_t) - 1:
                    z_trans = torch.zeros_like(x_0_trans)
                    z_rot = torch.zeros_like(x_0_rot)
                else:
                    z_trans = torch.randn_like(x_0_trans)
                    z_rot = torch.randn_like(x_0_rot)

                x_prev_trans = a_bar_prev.sqrt() * x_0_trans + ddim_coeffient * pred_link_trans_noise + sigma_t * z_trans * self.noise_lambda
                x_prev_rot = a_bar_prev.sqrt() * x_0_rot + ddim_coeffient * pred_link_rot_noise + sigma_t * z_rot * self.noise_lambda

                noisy_V_R_trans = x_prev_trans
                noisy_V_R_rot = x_prev_rot

                # update node and edge
                noisy_V_R = torch.cat([noisy_V_R_trans, noisy_V_R_rot, link_robot_embeds], dim=-1)
                noisy_V_R_pose = noisy_V_R[:, :, :6]
                noisy_V_R_se3 = vector_to_matrix(noisy_V_R_pose)                # [B*T, L, 4, 4] 
                noisy_E_RR = matrix_to_vector(                                  # [B*T, L, L, 6]
                    compute_batch_relative_se3(                       
                        noisy_V_R_se3, noisy_V_R_se3
                    )
                )
                object_positions = object_nodes[:, :, :3]
                B, P, _ = object_positions.shape
                object_se3 = torch.eye(4, device=device, dtype=dtype).expand(B, P, -1, -1).clone()
                object_se3[:, :, :3, 3] = object_positions                                                   
                noisy_E_OR = compute_batch_relative_se3(noisy_V_R_se3, object_se3)
                noisy_E_OR = matrix_to_vector(noisy_E_OR)

                # save snapshot
                pred_trans = noisy_V_R_trans * scale + centroids
                pred_rot = noisy_V_R_rot
                pred_pose = torch.cat([pred_trans, pred_rot], dim=-1)

                predict_link_pose_dict = {}
                denoised_step = ddim_t[i + 1].item() if i < len(ddim_t) - 1 else 0
                predict_link_pose = vector_to_matrix(pred_pose[:, :valid_links])       # [B, L, 4, 4]
                for link_id, link_name in enumerate(link_names):
                    predict_link_pose_dict[link_name] = predict_link_pose[:, link_id]
                all_diffuse_step_poses_dict[denoised_step] = predict_link_pose_dict
        return all_diffuse_step_poses_dict
