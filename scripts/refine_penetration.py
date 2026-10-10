"""Test-time finger refinement on the same TRO samples.

Uses object mesh signed distance (already available at inference) to reduce
penetration and keep contact, then re-runs Isaac Gym on the refined q.
"""
import argparse
import os
import sys

import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from validation.validate_utils import validate_isaac


def load_object_surface(object_name, n_pts=8192):
    dataset, name = object_name.split("+")
    mesh_path = os.path.join(
        ROOT_DIR, f"data/data_urdf/object/{dataset}/{name}/{name}.stl"
    )
    mesh = trimesh.load_mesh(mesh_path)
    pts, faces = mesh.sample(n_pts, return_index=True)
    normals = mesh.face_normals[faces]
    return (
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(normals, dtype=torch.float32),
    )


def signed_distance(robot_pc, obj_pts, obj_nrm):
    diff = robot_pc.unsqueeze(1) - obj_pts.unsqueeze(0)
    dist2 = (diff * diff).sum(-1)
    nn = dist2.argmin(dim=1)
    nearest = obj_pts[nn]
    normal = obj_nrm[nn]
    return ((robot_pc - nearest) * normal).sum(-1)


def refine_one(hand, q, obj_pts, obj_nrm, n_iter=40, lr=0.02):
    q = q.detach().cpu().float()
    root = q[:6].clone()
    fingers0 = q[6:].clone()
    fingers = fingers0.clone().requires_grad_(True)
    opt = torch.optim.Adam([fingers], lr=lr)
    lower, upper = hand.pk_chain.get_joint_limits()
    lower = torch.tensor(lower[6:], dtype=torch.float32)
    upper = torch.tensor(upper[6:], dtype=torch.float32)
    obj_pts = obj_pts.to(q.device)
    obj_nrm = obj_nrm.to(q.device)

    for _ in range(n_iter):
        opt.zero_grad()
        q_full = torch.cat([root, fingers])
        pcs, _ = hand.get_transformed_links_pc(q_full)
        robot = torch.cat(list(pcs.values()), dim=0)
        sd = signed_distance(robot, obj_pts, obj_nrm)
        pen = torch.relu(-sd - 0.001).mean()
        contact = torch.relu(sd - 0.006).mean()
        reg = (fingers - fingers0).pow(2).mean()
        loss = 8.0 * pen + 0.5 * contact + 0.2 * reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            fingers.clamp_(lower, upper)
    return torch.cat([root, fingers.detach()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis", required=True)
    parser.add_argument("--object_name", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    vis = torch.load(args.vis, map_location="cpu")
    device = torch.device("cpu")
    rows = vis
    if args.object_name:
        rows = [x for x in vis if x["object_name"] == args.object_name]

    for item in rows:
        robot_name = item["robot_name"]
        object_name = item["object_name"]
        q = item["predict_q"].float()
        base_sr = torch.as_tensor(item["success"]).float().mean().item()
        print(f"[{robot_name}/{object_name}] baseline_sr={base_sr:.3f} n={len(q)}")

        hand = create_hand_model(robot_name, device)
        obj_pts, obj_nrm = load_object_surface(object_name)
        refined = []
        for i in range(q.shape[0]):
            refined.append(refine_one(hand, q[i], obj_pts, obj_nrm))
            if (i + 1) % 20 == 0:
                print(f"  refined {i+1}/{len(q)}")
        q_ref = torch.stack(refined, dim=0)

        success, _ = validate_isaac(robot_name, object_name, q_ref, gpu=args.gpu)
        sr = torch.as_tensor(success).float().mean().item()
        print(f"[{robot_name}/{object_name}] refined_sr={sr:.3f}")
        out = os.path.join(os.path.dirname(args.vis), f"refined_{object_name.replace('+','-')}.pt")
        torch.save({"q": q_ref, "success": success, "baseline_sr": base_sr, "refined_sr": sr}, out)


if __name__ == "__main__":
    main()
