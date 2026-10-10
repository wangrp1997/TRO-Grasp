"""METHOD_V18 single-view depth occupancy: helpers (no TRO).

Contract (METHOD_V18 / I01 / I02):
- M_tgt = image-plane 4-neigh depth flood only (no 3D radius, no max_pix).
- L = L_clt + L_tgt + L_lim (joint limits + analytical self-collision; no L_cov).
- q0: wrist on click ray, open palm from URDF canonical.

Engineering (still image-plane z-buffer, not mesh SDF):
- wrist stays on click ray (t·ray); free rpy+joints
- mean-normalize L_clt/L_tgt; L_appr pulls hand to surface depth
- soft UV attract to click (recoverable when hard ov=0)
- multi-start + coarse→fine; soft finger-close for force closure
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image


def flood_fill_depth(
    D: np.ndarray,
    u: tuple[int, int],
    K: np.ndarray | None = None,
    thr_m: float = 0.02,
) -> np.ndarray:
    del K
    H, W = D.shape
    v0, u0 = int(u[0]), int(u[1])
    if not (0 <= v0 < H and 0 <= u0 < W) or D[v0, u0] <= 0:
        return np.zeros((H, W), dtype=bool)
    z_seed = float(D[v0, u0])
    out = np.zeros((H, W), dtype=bool)
    stack = [(v0, u0)]
    out[v0, u0] = True
    while stack:
        v, uu = stack.pop()
        z = float(D[v, uu])
        for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nv, nu = v + dv, uu + du
            if not (0 <= nv < H and 0 <= nu < W) or out[nv, nu]:
                continue
            zn = float(D[nv, nu])
            if zn <= 0 or abs(zn - z) >= thr_m or abs(zn - z_seed) >= thr_m:
                continue
            out[nv, nu] = True
            stack.append((nv, nu))
    return out


def parse_urdf_joint_limits(urdf_path: str | Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(str(urdf_path)).getroot()
    out: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        jtype = joint.get("type", "")
        if jtype not in ("revolute", "prismatic"):
            continue
        name = joint.get("name")
        lim = joint.find("limit")
        if name is None or lim is None:
            continue
        lo, hi = lim.get("lower"), lim.get("upper")
        if lo is None or hi is None:
            continue
        out[name] = (float(lo), float(hi))
    return out


def joint_limit_tensors(hand, device=None, dtype=torch.float32):
    device = device or hand.device
    names = list(hand.pk_chain.get_joint_parameter_names())
    parsed = parse_urdf_joint_limits(hand.urdf_path)
    pk_lo, pk_hi = hand.pk_chain.get_joint_limits()
    lo, hi = [], []
    for i, name in enumerate(names):
        if name in parsed:
            a, b = parsed[name]
        else:
            a, b = float(pk_lo[i]), float(pk_hi[i])
        lo.append(a)
        hi.append(b)
    return (
        torch.tensor(lo, device=device, dtype=dtype),
        torch.tensor(hi, device=device, dtype=dtype),
    )


def limit_loss(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (F.relu(lower - q) + F.relu(q - upper)).sum()


def load_frame(extracted: Path, scene: str, fid: str, obj_index: int = 0):
    cam = extracted / "scenes" / scene / "realsense"
    meta = sio.loadmat(str(cam / "meta" / f"{fid}.mat"))
    factor = float(np.array(meta["factor_depth"]).reshape(-1)[0])
    K = np.array(meta["intrinsic_matrix"], dtype=np.float64)
    depth = np.array(Image.open(cam / "depth_gt" / f"{fid}.png"))
    D = (depth.astype(np.float64) / factor).astype(np.float32)
    tree = ET.parse(cam / "annotations" / f"{fid}.xml")
    objs = list(tree.getroot().iter("obj"))
    o = objs[obj_index]
    oid = int(o.findtext("obj_id"))
    label = np.array(Image.open(cam / "label_gt" / f"{fid}.png"))
    gt = label == (oid + 1)
    ys, xs = np.where(gt)
    if len(ys) == 0:
        raise RuntimeError(f"empty GT for {scene}/{fid} obj {oid}")
    v, u = int(ys.mean()), int(xs.mean())
    if D[v, u] <= 0:
        d2 = (ys - v) ** 2 + (xs - u) ** 2
        for i in np.argsort(d2):
            if D[ys[i], xs[i]] > 0:
                v, u = int(ys[i]), int(xs[i])
                break
    return {
        "D": D,
        "K": K.astype(np.float32),
        "click": (v, u),
        "obj_name": o.findtext("obj_name"),
        "obj_id": oid,
        "gt": gt,
        "fid": fid,
        "scene": scene,
    }


def _raster_meshes(hand, max_faces_per_link: int = 4000):
    cached = getattr(hand, "_v18_raster_meshes", None)
    if cached is not None:
        return cached
    import trimesh

    out = {}
    for link_name, mesh in hand.meshes.items():
        if link_name not in hand.vertices:
            continue
        m = mesh
        if hasattr(m, "faces") and len(m.faces) > max_faces_per_link:
            try:
                m = m.simplify_quadratic_decimation(max_faces_per_link)
            except Exception:
                idx = np.linspace(0, len(m.faces) - 1, max_faces_per_link).astype(np.int64)
                m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[idx], process=False)
        out[link_name] = m
    hand._v18_raster_meshes = out
    return out


def hand_verts_faces(hand, q: torch.Tensor):
    hand.update_status(q)
    meshes = _raster_meshes(hand)
    verts_list, faces_list, off = [], [], 0
    for link_name, mesh in meshes.items():
        V = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=q.device)
        F = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64, device=q.device)
        T = hand.frame_status[link_name].get_matrix()[0]
        ones = torch.ones((V.shape[0], 1), device=q.device, dtype=V.dtype)
        Vh = torch.cat([V, ones], dim=1) @ T.T
        verts_list.append(Vh[:, :3])
        faces_list.append(F + off)
        off += V.shape[0]
    if not verts_list:
        raise RuntimeError(f"no hand mesh verts for {getattr(hand, 'robot_name', '?')}")
    return torch.cat(verts_list, dim=0), torch.cat(faces_list, dim=0)


_GLCTX = None


def raster_depth(verts, faces, K, H, W, near=0.05, far=2.0):
    import nvdiffrast.torch as dr

    global _GLCTX
    device = verts.device
    if _GLCTX is None:
        _GLCTX = dr.RasterizeCudaContext(device=device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2].clamp(min=1e-4)
    px = fx * x / z + cx
    py = fy * y / z + cy
    x_ndc = 2.0 * px / W - 1.0
    y_ndc = 1.0 - 2.0 * py / H
    z_ndc = (z - near) / (far - near) * 2.0 - 1.0
    pos = torch.stack([x_ndc, y_ndc, z_ndc, torch.ones_like(z)], dim=-1)[None]
    rast, _ = dr.rasterize(_GLCTX, pos.contiguous(), faces.int().contiguous(), resolution=[H, W])
    z_img, _ = dr.interpolate(z[None, :, None], rast, faces.int().contiguous())
    mask = rast[..., 3] > 0
    depth = torch.where(mask, z_img[..., 0], torch.zeros_like(z_img[..., 0]))
    return depth[0], mask[0]


def image_losses(D, M_tgt, M_clt, depth_hat, hand_mask, dmin=0.0, dmax=0.015):
    clt = M_clt & (D > 0) & (depth_hat > 0)
    n_clt = clt.float().sum().clamp(min=1.0)
    L_clt = F.relu(D - depth_hat)[clt].sum() / n_clt
    both = M_tgt & hand_mask & (D > 0) & (depth_hat > 0)
    n_both = both.float().sum().clamp(min=1.0)
    d = (D - depth_hat).abs()
    band = (F.relu(dmin - d) + F.relu(d - dmax))[both].sum() / n_both
    # miss: target pixels without hand → outside contact band (image-plane)
    miss = (M_tgt & (D > 0) & (~hand_mask)).float().mean()
    L_tgt = band + miss * dmax
    return L_clt, L_tgt


def _approach_loss(D, M_tgt, depth_hat, hand_mask):
    both = M_tgt & hand_mask & (D > 0) & (depth_hat > 0)
    n = both.float().sum()
    if n < 1:
        return D.new_zeros(())
    return F.relu(depth_hat - D)[both].sum() / n


# DGN2 SimulationEvaluator canonical_frame_rotation; gripper approach = (R @ C.T)[:, 0]
_DGN2_CANONICAL_FRAME = torch.tensor(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32
)


def _euler_xyz_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """XYZ intrinsic euler → 3×3 (matches scipy Rotation XYZ / TRO floating q)."""
    rx, ry, rz = rpy[0], rpy[1], rpy[2]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    one = rpy.new_ones(())
    zero = rpy.new_zeros(())
    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, -sx]),
            torch.stack([zero, sx, cx]),
        ]
    )
    Ry = torch.stack(
        [
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, zero]),
            torch.stack([sz, cz, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    return Rz @ Ry @ Rx


def dgn2_gripper_approach_from_rpy(rpy: torch.Tensor) -> torch.Tensor:
    """Camera/table-frame DGN2 gripper approach axis from floating-base rpy."""
    C = _DGN2_CANONICAL_FRAME.to(device=rpy.device, dtype=rpy.dtype)
    R = _euler_xyz_to_matrix(rpy)
    return (R @ C.T)[:, 0]


def _dir_approach_loss(rpy: torch.Tensor, prefer_dir: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Penalize approach · prefer_dir < margin (margin=cos60° for DGN2 top grasps).

    Squared hinge only; for near-vertical top grasps use margin≳0.9 (DexClutterBench).
    """
    approach = dgn2_gripper_approach_from_rpy(rpy)
    prefer = prefer_dir / (prefer_dir.norm() + 1e-8)
    return F.relu(margin - (approach * prefer).sum()) ** 2


def _matrix_to_euler_xyz(R1: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(R1[0, 0] ** 2 + R1[1, 0] ** 2).clamp(min=1e-8)
    x = torch.atan2(R1[2, 1], R1[2, 2])
    y = torch.atan2(-R1[2, 0], sy)
    z = torch.atan2(R1[1, 0], R1[0, 0])
    return torch.stack([x, y, z])


def _snap_rpy_approach_to_prefer(rpy: torch.Tensor, prefer_dir: torch.Tensor) -> torch.Tensor:
    """Rotate rpy so DGN2 gripper approach aligns with prefer_dir (keep twist free)."""
    approach = dgn2_gripper_approach_from_rpy(rpy)
    prefer = prefer_dir / (prefer_dir.norm() + 1e-8)
    a = approach / (approach.norm() + 1e-8)
    # Rodrigues: rotate a → prefer
    v = torch.linalg.cross(a, prefer)
    c = (a * prefer).sum().clamp(-1.0, 1.0)
    s = v.norm()
    if float(s.detach()) < 1e-6:
        return rpy
    vx = torch.stack(
        [
            torch.stack([rpy.new_zeros(()), -v[2], v[1]]),
            torch.stack([v[2], rpy.new_zeros(()), -v[0]]),
            torch.stack([-v[1], v[0], rpy.new_zeros(())]),
        ]
    )
    R_delta = (
        torch.eye(3, device=rpy.device, dtype=rpy.dtype)
        + vx
        + vx @ vx * ((1.0 - c) / (s * s + 1e-8))
    )
    R0 = _euler_xyz_to_matrix(rpy)
    return _matrix_to_euler_xyz(R_delta @ R0)


def _mask_pca_axis_cam(M_tgt: np.ndarray, D: np.ndarray, K: np.ndarray, prefer: np.ndarray) -> np.ndarray | None:
    """Major axis of M_tgt in cam, projected onto plane ⟂ prefer (depth flood only)."""
    ys, xs = np.where(M_tgt)
    if ys.size < 16:
        return None
    # subsample
    step = max(1, ys.size // 400)
    ys, xs = ys[::step], xs[::step]
    z = D[ys, xs].astype(np.float64)
    ok = z > 0
    if ok.sum() < 16:
        return None
    ys, xs, z = ys[ok], xs[ok], z[ok]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = pts.T @ pts / max(pts.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    pref = prefer / (np.linalg.norm(prefer) + 1e-8)
    major = major - (major @ pref) * pref
    n = np.linalg.norm(major)
    if n < 1e-6:
        return None
    return (major / n).astype(np.float64)


def _rpy_from_approach_and_y(approach: torch.Tensor, y_axis: torch.Tensor) -> torch.Tensor:
    """Build floating rpy from DGN2 approach + gripper-y (canonical frame columns)."""
    a = approach / (approach.norm() + 1e-8)
    y = y_axis - (y_axis * a).sum() * a
    y = y / (y.norm() + 1e-8)
    z = torch.linalg.cross(a, y)
    z = z / (z.norm() + 1e-8)
    y = torch.linalg.cross(z, a)
    # columns of (R @ C.T) are approach,y,z ⇒ R @ C.T = [a|y|z] ⇒ R = [a|y|z] @ C
    C = _DGN2_CANONICAL_FRAME.to(device=a.device, dtype=a.dtype)
    axes = torch.stack([a, y, z], dim=1)
    R = axes @ C
    return _matrix_to_euler_xyz(R)


def _uv_attract(verts, K, click_uv, H, W):
    """Soft image-plane pull of hand centroid toward click (differentiable)."""
    z = verts[:, 2].clamp(min=1e-3)
    u = K[0, 0] * verts[:, 0] / z + K[0, 2]
    v = K[1, 1] * verts[:, 1] / z + K[1, 2]
    cu = u.mean()
    cv = v.mean()
    # normalize by image size
    return ((cu - click_uv[1]) / max(W, 1)) ** 2 + ((cv - click_uv[0]) / max(H, 1)) ** 2


def _tgt_contour_uv(M_tgt: np.ndarray, max_pts: int = 64) -> np.ndarray:
    """Image-plane contour samples of M_tgt (4-neigh boundary)."""
    M = M_tgt.astype(bool)
    if M.sum() == 0:
        return np.zeros((0, 2), np.float32)
    # boundary: in mask but has out-of-mask 4-neighbor
    up = np.zeros_like(M)
    down = np.zeros_like(M)
    left = np.zeros_like(M)
    right = np.zeros_like(M)
    up[1:] = M[:-1]
    down[:-1] = M[1:]
    left[:, 1:] = M[:, :-1]
    right[:, :-1] = M[:, 1:]
    border = M & ~(up & down & left & right)
    vs, us = np.where(border)
    if len(vs) == 0:
        vs, us = np.where(M)
    if len(vs) > max_pts:
        idx = np.linspace(0, len(vs) - 1, max_pts).astype(int)
        vs, us = vs[idx], us[idx]
    return np.stack([vs.astype(np.float32), us.astype(np.float32)], axis=1)


def _contour_attract(verts, K, contour_uv, H, W):
    """Pull a few hand verts toward target silhouette in image plane (not 3D SDF)."""
    if contour_uv is None or contour_uv.numel() == 0:
        return verts.new_zeros(())
    z = verts[:, 2].clamp(min=1e-3)
    u = K[0, 0] * verts[:, 0] / z + K[0, 2]
    v = K[1, 1] * verts[:, 1] / z + K[1, 2]
    # subsample hand verts
    n = min(48, verts.shape[0])
    idx = torch.linspace(0, verts.shape[0] - 1, n).long().to(verts.device)
    pu = torch.stack([v[idx], u[idx]], dim=-1)  # (n,2) as (v,u)
    # contour (m,2)
    c = contour_uv.to(device=verts.device, dtype=verts.dtype)
    # normalize
    scale = torch.tensor([float(H), float(W)], device=verts.device, dtype=verts.dtype).clamp(min=1.0)
    pu = pu / scale
    c = c / scale
    d = torch.cdist(pu, c).min(dim=1).values.mean()
    return d


def self_collision_loss(hand, q: torch.Tensor, margin: float = 0.002) -> torch.Tensor:
    if hand.links_pc is None or len(hand.links_pc) < 2:
        return q.new_zeros(())
    pcs, _ = hand.get_transformed_links_pc(q)
    names = [n for n in pcs.keys() if n in hand.vertices]
    if len(names) < 2:
        return q.new_zeros(())
    pts = []
    for n in names:
        p = pcs[n]
        if p.shape[0] > 64:
            idx = torch.linspace(0, p.shape[0] - 1, 64).long()
            p = p[idx]
        pts.append(p)
    loss = q.new_zeros(())
    for i in range(len(pts)):
        for j in range(i + 2, len(pts)):
            d = torch.cdist(pts[i], pts[j]).min()
            loss = loss + F.relu(margin - d)
    return loss


def _init_candidates(
    hand, q_can, ray_t, z, Ks, Hs, Ws, Mt, device, lower, upper, topk: int = 6, *, K_full=None, click=None, D=None,
    approach_prefer_dir=None,
    wrist_above_m: float = 0.0,
):
    """q0 on click/contour rays + open/semi-open palm; top-K by overlap.

    If ``approach_prefer_dir`` is set (cam-frame unit-ish), boost seeds whose
    DGN2 gripper approach aligns with it (DexClutterBench table-down via E^T g).
    When ``wrist_above_m`` > 0, also seed wrists offset opposite prefer_dir so
    top-down contact is optimized above the surface (not sunk to table z≈0).
    """
    # Moderate close (~DGN2 LEAP scale), not near-upper slam (abs_mean≈1.3 kills bite after relax).
    finger_sets = [
        q_can[6:].detach(),
        (0.45 * lower[6:] + 0.55 * upper[6:]).detach(),
        (0.55 * q_can[6:] + 0.45 * (0.35 * lower[6:] + 0.65 * upper[6:])).detach(),
    ]
    prefer_np = None
    if approach_prefer_dir is not None:
        prefer_np = np.asarray(approach_prefer_dir, dtype=np.float64).reshape(3)
        n = np.linalg.norm(prefer_np)
        if n > 1e-8:
            prefer_np = prefer_np / n
        else:
            prefer_np = None

    # Fast path for top-down + wrist_above: skip rpy×ray combinatorial explosion.
    if prefer_np is not None and float(wrist_above_m) > 0 and D is not None and K_full is not None and click is not None:
        wa = float(wrist_above_m)
        pref_t = torch.as_tensor(prefer_np, device=device, dtype=torch.float32)
        contact = torch.tensor([z], device=device, dtype=torch.float32) * ray_t
        M_full = flood_fill_depth(D, click, K_full)
        maj = _mask_pca_axis_cam(M_full, D, K_full, prefer_np)
        y_axes = []
        if maj is not None:
            maj_t = torch.as_tensor(maj, device=device, dtype=torch.float32)
            perp = torch.linalg.cross(pref_t, maj_t)
            if float(perp.norm()) > 1e-6:
                perp = perp / (perp.norm() + 1e-8)
                y_axes = [maj_t, -maj_t, perp, -perp]
            else:
                y_axes = [maj_t, -maj_t]
        else:
            tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
            if abs(float((tmp * pref_t).sum())) > 0.9:
                tmp = torch.tensor([0.0, 1.0, 0.0], device=device)
            y0 = torch.linalg.cross(pref_t, tmp)
            y0 = y0 / (y0.norm() + 1e-8)
            y_axes = [y0, -y0]
        y0 = y_axes[0]
        z0 = torch.linalg.cross(pref_t, y0)
        z0 = z0 / (z0.norm() + 1e-8)
        for ang in (0.125 * math.pi, 0.25 * math.pi, 0.375 * math.pi, 0.5 * math.pi, 0.75 * math.pi):
            c, s = math.cos(ang), math.sin(ang)
            y_rot = c * y0 + s * z0
            y_axes.append(y_rot / (y_rot.norm() + 1e-8))
        scored = []
        with torch.no_grad():
            for ao in (0.85 * wa, wa, 1.15 * wa):
                xyz = contact - float(ao) * pref_t
                xyz = torch.stack(
                    [
                        xyz[0].clamp(-0.35, 0.35),
                        xyz[1].clamp(-0.35, 0.35),
                        xyz[2].clamp(0.08, 1.4),
                    ]
                )
                for y_ax in y_axes:
                    rpy = _rpy_from_approach_and_y(pref_t, y_ax)
                    for fingers in finger_sets[:2]:
                        rest = torch.cat(
                            [rpy, fingers.to(device=device, dtype=torch.float32)], dim=0
                        )
                        q = torch.cat([xyz, rest[: hand.dof - 3]], dim=0)
                        try:
                            verts, faces = hand_verts_faces(hand, q)
                            _, hand_mask = raster_depth(verts, faces, Ks, Hs, Ws)
                        except Exception:
                            continue
                        ov = int((hand_mask & Mt).sum().item())
                        sc = float(ov) + 80.0 * (1.0 - abs(ao - wa) / max(wa, 1e-3))
                        scored.append((sc, ov, xyz.detach().clone(), rest.detach().clone()))
        scored.sort(key=lambda x: -x[0])
        uniq = []
        seen = set()
        for sc, ov, xyz, rest in scored:
            key = tuple((rest[:3] / (math.pi / 4)).round().tolist()) + (
                int(round(float(xyz[2]) * 25)),
            )
            if key in seen:
                continue
            seen.add(key)
            uniq.append((max(int(ov), 0), xyz, rest))
            if len(uniq) >= topk:
                break
        if not uniq:
            xyz0 = (contact - wa * pref_t).detach().clone()
            rpy0 = _rpy_from_approach_and_y(pref_t, y_axes[0])
            rest0 = torch.cat([rpy0, q_can[6:].to(device)], dim=0)
            uniq = [(0, xyz0, rest0)]
        return uniq

    rpy_list = []
    for yaw in (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi, 0.25 * math.pi, -0.25 * math.pi):
        for pitch in (0.0, 0.5 * math.pi, -0.5 * math.pi, 0.4 * math.pi, -0.4 * math.pi):
            for roll in (0.0, 0.5 * math.pi, -0.5 * math.pi, 0.25 * math.pi):
                rpy_list.append([roll, pitch, yaw])
    rpy_list.append(q_can[3:6].detach().cpu().tolist())
    # rays: primary click + a few M_tgt contour seeds (still from D, no GT)
    rays = [ray_t]
    zs = [z]
    if K_full is not None and click is not None and D is not None:
        contour = _tgt_contour_uv(Mt.detach().cpu().numpy() if torch.is_tensor(Mt) else Mt, max_pts=12)
        # Mt here is downscaled; use full M via D flood already done outside — pass contour in full res below
    # build extra rays from full-res contour if provided via K_full/D/click
    if K_full is not None and D is not None and click is not None:
        M_full = flood_fill_depth(D, click, K_full)
        for vv, uu in _tgt_contour_uv(M_full, max_pts=10):
            vv, uu = int(vv), int(uu)
            if not (0 <= vv < D.shape[0] and 0 <= uu < D.shape[1]) or D[vv, uu] <= 0:
                continue
            zz = float(D[vv, uu])
            rr = np.array(
                [(uu - K_full[0, 2]) / K_full[0, 0], (vv - K_full[1, 2]) / K_full[1, 1], 1.0],
                np.float32,
            )
            rr /= np.linalg.norm(rr) + 1e-8
            rays.append(torch.as_tensor(rr, device=device, dtype=torch.float32))
            zs.append(zz)
    # Retreat along ray from click depth (m); prefer closer seeds for wrist reach.
    t_fracs = (0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06)
    scored = []
    with torch.no_grad():
        for ray_i, zi in zip(rays, zs):
            t_cands = [max(0.08, zi - d) for d in t_fracs]
            t_cands.append(max(0.08, zi - 0.012))
            for td in t_cands:
                xyz = torch.tensor([td], device=device, dtype=torch.float32) * ray_i
                for rpy in rpy_list:
                    for fingers in finger_sets[:2]:
                        rest = torch.cat(
                            [
                                torch.tensor(rpy, device=device, dtype=torch.float32),
                                fingers.to(device=device, dtype=torch.float32),
                            ],
                            dim=0,
                        )
                        q = torch.cat([xyz, rest[: hand.dof - 3]], dim=0)
                        try:
                            verts, faces = hand_verts_faces(hand, q)
                            _, hand_mask = raster_depth(verts, faces, Ks, Hs, Ws)
                        except Exception:
                            continue
                        ov = int((hand_mask & Mt).sum().item())
                        if ov < 8:
                            continue
                        # Higher score = more overlap + wrist nearer surface depth.
                        prox = td / max(zi, 1e-3)
                        sc = ov + 120.0 * prox - 40.0 * max(0.0, (zi - td) - 0.04)
                        if prefer_np is not None:
                            rpy_t = torch.tensor(rpy, device=device, dtype=torch.float32)
                            ap = dgn2_gripper_approach_from_rpy(rpy_t).detach().cpu().numpy()
                            sc += 180.0 * float(np.dot(ap, prefer_np))
                        scored.append(
                            (
                                sc,
                                ov,
                                xyz.detach().clone(),
                                rest.detach().clone(),
                            )
                        )
        # Compact top-down wrist seeds (primary ray only): contact − ao*prefer.
        # Keep this small — full grid × ao exploded runtime.
        if prefer_np is not None and float(wrist_above_m) > 0:
            wa = float(wrist_above_m)
            pref_t = torch.as_tensor(prefer_np, device=device, dtype=torch.float32)
            contact = torch.tensor([z], device=device, dtype=torch.float32) * ray_t
            # Prefer rpy already aligned with gravity (cheap filter, no raster yet).
            # Cheap vertical seeds: snap a sparse yaw grid (full snap×rpy is too slow).
            down_rpy = []
            for rpy in rpy_list[::10]:
                rpy_t = torch.tensor(rpy, device=device, dtype=torch.float32)
                down_rpy.append(
                    _snap_rpy_approach_to_prefer(rpy_t, pref_t).detach().cpu().tolist()
                )
            for ao in (0.7 * wa, wa, 1.2 * wa):
                xyz = contact - float(ao) * pref_t
                xyz = torch.stack(
                    [
                        xyz[0].clamp(-0.35, 0.35),
                        xyz[1].clamp(-0.35, 0.35),
                        xyz[2].clamp(0.08, 1.4),
                    ]
                )
                for rpy in down_rpy:
                    for fingers in finger_sets[:2]:
                        rest = torch.cat(
                            [
                                torch.tensor(rpy, device=device, dtype=torch.float32),
                                fingers.to(device=device, dtype=torch.float32),
                            ],
                            dim=0,
                        )
                        q = torch.cat([xyz, rest[: hand.dof - 3]], dim=0)
                        try:
                            verts, faces = hand_verts_faces(hand, q)
                            _, hand_mask = raster_depth(verts, faces, Ks, Hs, Ws)
                        except Exception:
                            continue
                        ov = int((hand_mask & Mt).sum().item())
                        if ov < 4:
                            continue
                        rpy_t = torch.tensor(rpy, device=device, dtype=torch.float32)
                        ap = dgn2_gripper_approach_from_rpy(rpy_t).detach().cpu().numpy()
                        sc = ov + 180.0 * float(np.dot(ap, prefer_np)) + 80.0 * (
                            1.0 - abs(ao - wa) / max(wa, 1e-3)
                        )
                        scored.append(
                            (
                                sc,
                                ov,
                                xyz.detach().clone(),
                                rest.detach().clone(),
                            )
                        )
    scored.sort(key=lambda x: -x[0])
    uniq = []
    seen = set()
    for sc, ov, xyz, rest in scored:
        key = tuple((rest[:3] / (math.pi / 4)).round().tolist()) + (
            int(round(float(xyz[2]) * 25)),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append((ov, xyz, rest))
        if len(uniq) >= topk:
            break
    if not uniq:
        xyz0 = (max(0.08, z - 0.018) * ray_t).detach().clone()
        uniq = [(0, xyz0, q_can[3:].detach().clone())]
    return uniq


def optimize_frame(
    hand,
    D: np.ndarray,
    K: np.ndarray,
    click: tuple[int, int],
    *,
    steps: int = 120,
    lr: float = 0.02,
    scale: int = 2,
    device=None,
    w_lim: float = 1.0,
    w_self: float = 0.15,
    w_appr: float = 24.0,
    w_tgt: float = 8.0,
    w_clt: float = 2.0,
    w_uv: float = 2.0,
    w_contour: float = 3.0,
    n_starts: int = 5,
    return_all: bool = False,
    approach_prefer_dir=None,
    w_dir: float = 12.0,
    dir_margin: float = 0.5,
    wrist_above_m: float = 0.0,
    w_wrist_above: float = 40.0,
):
    """q0 on ray; free SE(3)+joints with soft ray prior; image-plane z-buffer L.

    ``approach_prefer_dir``: optional cam-frame direction for DGN2 gripper approach
    (DexClutterBench: E[:3,:3].T @ [0,0,-1] so post cam→table approach points down).
    ``wrist_above_m``: when prefer_dir set, keep wrist ~this many meters opposite
    prefer_dir from the click depth point (top-down bite; avoids table-sunk z≈0.02).
    """
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    M_tgt = flood_fill_depth(D, click, K)
    M_clt = (D > 0) & (~M_tgt)
    v, u = click
    z = float(D[v, u]) if D[v, u] > 0 else 0.5
    ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0], np.float32)
    ray /= np.linalg.norm(ray)

    q_can = hand.get_canonical_q().to(device=device, dtype=torch.float32)
    lower, upper = joint_limit_tensors(hand, device=device)
    ray_t = torch.as_tensor(ray, device=device, dtype=torch.float32)
    click_uv = torch.tensor([float(v), float(u)], device=device, dtype=torch.float32)
    z_t = torch.tensor(z, device=device, dtype=torch.float32)
    contour_np = _tgt_contour_uv(M_tgt, max_pts=64)
    # large targets need wrap; scale contour weight by mask area
    area = float(M_tgt.sum())
    w_c_eff = w_contour * (1.0 + min(3.0, area / 5000.0))
    prefer_t = None
    if approach_prefer_dir is not None:
        prefer_t = torch.as_tensor(
            np.asarray(approach_prefer_dir, dtype=np.float32).reshape(3),
            device=device,
            dtype=torch.float32,
        )
        prefer_t = prefer_t / (prefer_t.norm() + 1e-8)
    # Default top-down wrist standoff when prefer_dir is active (DGN2 success ~0.14 m above table).
    wa_m = float(wrist_above_m)
    if prefer_t is not None and wa_m <= 0:
        wa_m = 0.12
    contact_t = (z_t * ray_t).detach()
    twist_axes = []
    if prefer_t is not None:
        maj = _mask_pca_axis_cam(M_tgt, D, K, prefer_t.detach().cpu().numpy())
        if maj is not None:
            maj_t = torch.as_tensor(maj, device=device, dtype=torch.float32)
            # ±major and a perpendicular in the table-parallel plane
            pref_np = prefer_t.detach()
            perp = torch.linalg.cross(pref_np, maj_t)
            if float(perp.norm()) > 1e-6:
                perp = perp / (perp.norm() + 1e-8)
                twist_axes = [maj_t, -maj_t, perp, -perp]
            else:
                twist_axes = [maj_t, -maj_t]

    def _run(scale_i, steps_i, xyz0, rest0, lr_i, close_w: float, se3_scale: float, twist_y=None):
        Hs, Ws = D.shape[0] // scale_i, D.shape[1] // scale_i
        Ds = torch.as_tensor(D[::scale_i, ::scale_i], device=device)
        Mt = torch.as_tensor(M_tgt[::scale_i, ::scale_i], device=device)
        Mc = torch.as_tensor(M_clt[::scale_i, ::scale_i], device=device)
        Ks = torch.as_tensor(K, device=device).clone()
        Ks[0, 0] /= scale_i
        Ks[1, 1] /= scale_i
        Ks[0, 2] /= scale_i
        Ks[1, 2] /= scale_i
        click_s = click_uv.clone()
        click_s[0] /= scale_i
        click_s[1] /= scale_i
        if contour_np.shape[0] > 0:
            c_s = torch.as_tensor(contour_np, device=device, dtype=torch.float32).clone()
            c_s[:, 0] /= scale_i
            c_s[:, 1] /= scale_i
        else:
            c_s = torch.zeros((0, 2), device=device, dtype=torch.float32)
        xyz_p = xyz0.clone().detach().requires_grad_(True)
        rest_init = rest0.clone().detach()
        if prefer_t is not None and rest_init.numel() >= 3:
            # Snap approach (+ optional PCA twist) so contact opt runs under top-down bite.
            with torch.no_grad():
                if twist_y is not None:
                    rest_init[:3] = _rpy_from_approach_and_y(prefer_t, twist_y)
                else:
                    rest_init[:3] = _snap_rpy_approach_to_prefer(rest_init[:3], prefer_t)
        q_rest = rest_init.clone().requires_grad_(True)
        opt = torch.optim.Adam(
            [
                {"params": [xyz_p], "lr": lr_i * se3_scale},
                {"params": [q_rest], "lr": lr_i},
            ]
        )
        # Soft close toward mid-upper; slam-to-0.85*upper overcloses LEAP vs DGN2 (~0.4 abs_mean).
        finger_hi = 0.40 * lower[6:] + 0.60 * upper[6:]
        hist = []
        last_L = last_appr = last_tgt = 0.0
        last_ov = 0.0
        for i in range(steps_i):
            opt.zero_grad()
            xyz = torch.stack(
                [
                    xyz_p[0].clamp(-0.35, 0.35),
                    xyz_p[1].clamp(-0.35, 0.35),
                    xyz_p[2].clamp(0.08, 1.4),
                ]
            )
            q = torch.cat([xyz, q_rest[: hand.dof - 3]], dim=0) if hand.dof > 3 else xyz
            verts, faces = hand_verts_faces(hand, q)
            depth_hat, hand_mask = raster_depth(verts, faces, Ks, Hs, Ws)
            frac = i / max(1, steps_i - 1)
            dmax = 0.012 * (1.0 - frac) + 0.0015 * frac
            L_clt, L_tgt = image_losses(Ds, Mt, Mc, depth_hat, hand_mask, dmax=dmax)
            L_appr = _approach_loss(Ds, Mt, depth_hat, hand_mask)
            L_uv = _uv_attract(verts, Ks, click_s, Hs, Ws)
            L_ctr = _contour_attract(verts, Ks, c_s, Hs, Ws)
            L_lim = limit_loss(q, lower[: q.numel()], upper[: q.numel()])
            L_self = self_collision_loss(hand, q) if w_self > 0 else q.new_zeros(())
            # When optimizing top-down (prefer_dir), drop hard ray stick so wrist can sit above.
            if prefer_t is not None and wa_m > 0:
                L_ray = q.new_zeros(())
                L_front = q.new_zeros(())
                L_back = q.new_zeros(())
                # wrist ≈ contact − wrist_above * prefer (prefer = gravity/into-table in cam).
                desired = contact_t - float(wa_m) * prefer_t
                L_wrist = ((xyz - desired) ** 2).sum() * float(w_wrist_above)
                # Soft band: above ∈ [0.6, 1.4] * wa_m along prefer.
                above = ((contact_t - xyz) * prefer_t).sum()
                L_wrist = L_wrist + F.relu(0.6 * wa_m - above) ** 2 * float(w_wrist_above)
                L_wrist = L_wrist + F.relu(above - 1.4 * wa_m) ** 2 * (0.5 * float(w_wrist_above))
            else:
                proj = (xyz * ray_t).sum() * ray_t
                L_ray = ((xyz - proj) ** 2).sum() * 0.03
                t_along = (xyz * ray_t).sum()
                # Stay just in front of flood depth; penalize floating too far back along ray.
                z_anchor = z_t - 0.012 * (1.0 - 0.35 * frac)
                L_front = F.relu(t_along - (z_t - 0.004)) ** 2 * 18.0
                L_back = F.relu(z_anchor - t_along) ** 2 * 22.0
                L_wrist = q.new_zeros(())
            if hand.dof > 6:
                L_close = ((q[6:] - finger_hi.to(q.device)) ** 2).mean() * close_w * (0.10 + 0.7 * frac)
            else:
                L_close = q.new_zeros(())
            if prefer_t is not None and q.numel() > 6:
                L_dir = _dir_approach_loss(q[3:6], prefer_t, margin=float(dir_margin)) * float(w_dir)
            else:
                L_dir = q.new_zeros(())
            ov = (hand_mask & Mt).float().sum()
            L = (
                w_clt * L_clt
                + w_tgt * L_tgt
                + w_appr * L_appr
                + w_uv * L_uv
                + w_c_eff * L_ctr
                + w_lim * L_lim
                + w_self * L_self
                + L_close
                + L_dir
                + L_ray
                + L_front
                + L_back
                + L_wrist
            )
            L.backward()
            opt.step()
            last_L = float(L.detach())
            last_appr = float(L_appr.detach())
            last_tgt = float(L_tgt.detach())
            last_ov = float(ov.detach())
            if i % 20 == 0 or i == steps_i - 1:
                hist.append(
                    {
                        "step": i,
                        "L": last_L,
                        "L_clt": float(L_clt),
                        "L_tgt": last_tgt,
                        "L_appr": last_appr,
                        "L_uv": float(L_uv),
                        "L_lim": float(L_lim),
                        "L_self": float(L_self),
                        "dmax": float(dmax),
                        "hand_pix": int(hand_mask.sum()),
                        "ov": int(last_ov),
                        "xyz": [float(x) for x in xyz.detach()],
                    }
                )
        xyz_f = xyz_p.detach().clone()
        xyz_f[0] = xyz_f[0].clamp(-0.35, 0.35)
        xyz_f[1] = xyz_f[1].clamp(-0.35, 0.35)
        xyz_f[2] = xyz_f[2].clamp(0.08, 1.4)
        q_final = torch.cat([xyz_f, q_rest.detach()[: hand.dof - 3]], 0)
        score = last_appr + last_tgt - 0.005 * last_ov + 0.05 * last_L
        if last_ov < 1:
            score += 100.0
        return q_final, xyz_f, q_rest.detach(), hist, score

    Hs0 = D.shape[0] // 4
    Ws0 = D.shape[1] // 4
    Mt0 = torch.as_tensor(M_tgt[::4, ::4], device=device)
    Ks0 = torch.as_tensor(K, device=device).clone()
    Ks0[0, 0] /= 4
    Ks0[1, 1] /= 4
    Ks0[0, 2] /= 4
    Ks0[1, 2] /= 4
    cands = _init_candidates(
        hand,
        q_can,
        ray_t,
        z,
        Ks0,
        Hs0,
        Ws0,
        Mt0,
        device,
        lower,
        upper,
        topk=n_starts,
        K_full=K,
        click=click,
        D=D,
        approach_prefer_dir=approach_prefer_dir,
        wrist_above_m=wa_m if prefer_t is not None else 0.0,
    )
    best = None
    all_packs = []
    # Milder close schedule under top-down prefer (keep DGN2-like finger scale).
    cw0, cw1, cw2 = (0.2, 0.45, 0.8) if prefer_t is not None else (0.3, 0.7, 1.5)
    # Pair each spatial seed with a PCA twist (cycle); covers thin objects like screwdrivers.
    twists = twist_axes if twist_axes else [None]
    run_list = []
    for i, (ov0, xyz0, rest0) in enumerate(cands):
        run_list.append((ov0, xyz0, rest0, twists[i % len(twists)]))
    # Extra: best spatial seed × all twists (cheap for thin-object azimuth).
    if twist_axes and cands:
        ov0, xyz0, rest0 = cands[0]
        for ty in twist_axes:
            run_list.append((ov0, xyz0, rest0, ty))
    for ov0, xyz0, rest0, twist_y in run_list:
        _, xyz1, rest1, hist1, _ = _run(
            4, max(40, steps // 3), xyz0, rest0, lr, close_w=cw0, se3_scale=1.0, twist_y=twist_y
        )
        _, xyz2, rest2, hist2, _ = _run(
            scale, steps, xyz1, rest1, lr * 0.55, close_w=cw1, se3_scale=0.8, twist_y=twist_y
        )
        q3, xyz3, rest3, hist3, score = _run(
            scale, max(40, steps // 3), xyz2, rest2, lr * 0.3, close_w=cw2, se3_scale=0.15, twist_y=twist_y
        )
        # Final bite: sink a few cm along prefer + close more (pre already cleared by high wrist).
        if prefer_t is not None and wa_m > 0:
            sink = min(0.035, 0.3 * wa_m)
            xyz_bite = (xyz3 + sink * prefer_t).detach().clone()
            xyz_bite[0] = xyz_bite[0].clamp(-0.35, 0.35)
            xyz_bite[1] = xyz_bite[1].clamp(-0.35, 0.35)
            xyz_bite[2] = xyz_bite[2].clamp(0.08, 1.4)
            # Temporarily tighten wrist band around the sunk pose.
            wa_saved = wa_m
            wa_m = max(0.06, wa_m - sink)
            q3, xyz3, rest3, hist4, score = _run(
                scale,
                max(30, steps // 4),
                xyz_bite,
                rest3,
                lr * 0.25,
                close_w=cw2 * 1.8,
                se3_scale=0.4,
                twist_y=twist_y,
            )
            wa_m = wa_saved
            hist3 = hist3 + hist4
            # Finger-only refine: freeze SE(3)/rpy; reset fingers to mild mid then wrap.
            # PRIV diag: wh5 pose + DGN2 fingers lifts; ours fingers on same pose do not.
            if hand.dof > 6:
                finger_mid = 0.50 * lower[6:] + 0.50 * upper[6:]
                rest_f = rest3.clone().detach()
                rest_f[3:] = (
                    0.55 * q_can[6:].to(rest_f.device)
                    + 0.45 * finger_mid.to(rest_f.device)
                )
                # tiny se3 lr ≈ freeze wrist/rpy; fingers get normal lr via close_w
                q3, xyz3, rest3, hist5, score = _run(
                    scale,
                    max(40, steps // 3),
                    xyz3.detach(),
                    rest_f,
                    lr * 0.35,
                    close_w=2.2,
                    se3_scale=0.02,
                    twist_y=twist_y,
                )
                hist3 = hist3 + hist5
        pack = {
            "q": q3.cpu(),
            "score": score,
            "init_ov": int(ov0),
            "hist": hist1 + hist2 + hist3,
        }
        all_packs.append(pack)
        if best is None or score < best["score"]:
            best = pack
    all_packs.sort(key=lambda p: p["score"])
    out = {
        "q": best["q"],
        "score": best["score"],
        "M_tgt_n": int(M_tgt.sum()),
        "M_clt_n": int(M_clt.sum()),
        "init_overlap": int(best["init_ov"]),
        "hist": best["hist"],
        "M_tgt": M_tgt,
        "n_starts": len(cands),
    }
    if return_all:
        out["candidates"] = all_packs
    return out
