import os
import argparse
import torch
import wandb
from datetime import datetime
from omegaconf import OmegaConf
from dataset.CMapDataset import create_dataloader
from model.tro_graph import RobotGraph

def prepare_input(batch, device):

    robot_pc_initial = batch['robot_pc_initial']
    for batch_robot_pc_initial in robot_pc_initial:
        for link_name, link_pc in batch_robot_pc_initial.items():
            batch_robot_pc_initial[link_name] = link_pc.to(device)

    robot_pc_target = batch['robot_pc_target']
    for batch_robot_pc_target in robot_pc_target:
        for link_name, link_pc in batch_robot_pc_target.items():
            batch_robot_pc_target[link_name] = link_pc.to(device)

    batch['object_pc'] = batch['object_pc'].to(device)
    batch['object_pc_normal'] = batch['object_pc_normal'].to(device)
    if 'view_dir' in batch:
        batch['view_dir'] = batch['view_dir'].to(device)
    if 'scene_pc' in batch and torch.is_tensor(batch['scene_pc']):
        batch['scene_pc'] = batch['scene_pc'].to(device)
    if 'scene_label' in batch and torch.is_tensor(batch['scene_label']):
        batch['scene_label'] = batch['scene_label'].to(device)
    batch['initial_q'] = [x.to(device) for x in batch['initial_q']]
    batch['target_q'] = [x.to(device) for x in batch['target_q']]
    batch['initial_se3'] = [x.to(device) for x in batch['initial_se3']]
    batch['target_se3'] = [x.to(device) for x in batch['target_se3']]
    batch['initial_vec'] = [x.to(device) for x in batch['initial_vec']]
    batch['target_vec'] = [x.to(device) for x in batch['target_vec']]
    
    return batch


def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = config.train.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(
        project=config.train.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print("Building dataloader...")
    dataloader = create_dataloader(config.dataset, is_train=True)
    print("Building model...")
    model = RobotGraph(**config.model).to(device)
    if getattr(model, "use_residual", False):
        model.freeze_eps_base()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params {total_params}")
    wandb.config.update({"total_params": total_params})

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.lr_step,
        gamma=config.train.lr_gamma
    )

    if config.train.resume_from:
        ckpt = torch.load(config.train.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if getattr(model, "use_residual", False):
            model.freeze_eps_base()
        finetune = bool(config.train.get("finetune", False))
        if finetune:
            start_epoch = 0
            print(f"Finetune from {config.train.resume_from} (reset epoch to 0)")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"]
            print(f"Resumed from {config.train.resume_from} at epoch {start_epoch}")
    else:
        start_epoch = 0

    num_epochs = config.train.epochs
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_id, batch in enumerate(dataloader):

            batch = prepare_input(batch, device)
            loss_dict = model(batch)
            loss = loss_dict['loss_total']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            log_data = {k: v.item() for k, v in loss_dict.items()}
            log_data.update({"lr": scheduler.get_last_lr()[0]})
            wandb.log(log_data)
            if batch_id % 50 == 0:
                print(
                    f"epoch {epoch} batch {batch_id}/{len(dataloader)} "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch_avg_loss": avg_epoch_loss})
        print(f"Epoch {epoch} avg_loss {avg_epoch_loss:.4f}", flush=True)

        os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
        latest_path = os.path.join(save_dir, "ckpt", "latest.pth")
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(payload, latest_path)

        save_every = int(config.train.get("save_freq", config.train.save_interval))
        if (epoch + 1) % save_every == 0:
            os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
            ckpt_path = os.path.join(save_dir, "ckpt", f"{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()
            }, ckpt_path)
            wandb.save(ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    train(config)
