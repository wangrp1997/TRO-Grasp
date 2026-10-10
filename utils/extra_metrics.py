"""Extra grasp metrics beyond TRO Table I (SR / time / diversity).

plane_identifiability: Kanatani (1995) plane reliability at contacting links only.
wrench_sigma_min: Li-Sastry (1988) min eigenvalue of G G^T; not our claim.
"""
import os
import torch

from utils.kanatani_reliability import patch_plane_reliability

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_object_pc_normals(object_name, device):
    name = object_name.split("+")
    path = os.path.join(ROOT_DIR, f"data/PointCloud/object/{name[0]}/{name[1]}.pt")
    data = torch.load(path, map_location=device)
    return data[:, :3], data[:, 3:]


def _signed_depth(robot_pc, object_pc, normals):
    dist, idx = torch.cdist(robot_pc, object_pc).min(dim=-1)
    obj = object_pc[idx]
    n = normals[idx]
    signed = ((robot_pc - obj) * n).sum(-1)
    return signed, n, dist


def compute_grasp_metrics(hand, object_name, predict_q, target_pos, link_names, contact_mm=8.0):
    """
    :param predict_q: (B, dof)
    :param target_pos: (B, L, 3) IK targets
    :return dict of scalars (mean over batch)
    """
    device = predict_q.device
    object_pc, normals = _load_object_pc_normals(object_name, device)
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.tensor(lower, device=device, dtype=predict_q.dtype)
    upper = torch.tensor(upper, device=device, dtype=predict_q.dtype)
    limit_viol = ((predict_q < lower) | (predict_q > upper)).float().mean()

    pd_list = []
    cov_list = []
    ident_list = []
    plane_list = []
    ik_list = []
    margin = contact_mm / 1000.0

    for i in range(predict_q.shape[0]):
        q = predict_q[i]
        pc_dict, se3 = hand.get_transformed_links_pc(q)
        fk_pos = se3[:, :3, 3]
        if target_pos is not None:
            ik_list.append((fk_pos - target_pos[i, : fk_pos.shape[0]]).norm(dim=-1).mean())

        link_xyz = []
        link_n = []
        contact_xyz = []
        depths = []
        contacts = 0
        n_links = 0
        for name, pc in pc_dict.items():
            n_links += 1
            signed, n_pt, _ = _signed_depth(pc, object_pc, normals)
            dmin, j = signed.min(dim=0)
            depths.append(torch.relu(-dmin))
            cxyz = pc.mean(dim=0)
            cn = n_pt[j]
            link_xyz.append(cxyz)
            link_n.append(cn)
            if dmin.abs() < margin:
                contacts += 1
                contact_xyz.append(pc[j])
        pd_list.append(torch.stack(depths).mean() if depths else torch.tensor(0.0, device=device))
        cov_list.append(torch.tensor(contacts / max(n_links, 1), device=device, dtype=predict_q.dtype))

        P = torch.stack(link_xyz, dim=0)
        N = torch.stack(link_n, dim=0)
        N = N / N.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tau = torch.cross(P, N, dim=-1)
        G = torch.cat([N, tau], dim=-1)
        gg = G.T @ G
        evals = torch.linalg.eigvalsh(gg)
        ident_list.append(evals[0].clamp_min(0.0))
        if contact_xyz:
            cxyz = torch.stack(contact_xyz, dim=0).unsqueeze(0)
            ident_w, _ = patch_plane_reliability(
                object_pc.unsqueeze(0), cxyz, knn=min(32, object_pc.shape[0])
            )
            plane_list.append(ident_w.mean())
        else:
            plane_list.append(torch.zeros((), device=device, dtype=predict_q.dtype))

    def _m(xs):
        return torch.stack(xs).mean().item() if xs else float("nan")

    return {
        "penetration_m": _m(pd_list),
        "contact_coverage": _m(cov_list),
        "ik_residual_m": _m(ik_list),
        "limit_violation": float(limit_viol.item()),
        "contact_identifiability": _m(ident_list),
        "wrench_sigma_min": _m(ident_list),
        "plane_identifiability": _m(plane_list),
    }
