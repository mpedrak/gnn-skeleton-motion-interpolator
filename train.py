import torch
import os
import argparse
import time
import gc
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from src.dataset import GraphSkeletonDataset
from src.model import SkeletalMotionInterpolator
from src.calculate_loss import calculate_loss
from src.utils.various import load_configs, log_string, set_global_seed, set_worker_seed


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    filename = args.config

    config, constants = load_configs([filename, "constants"])
    print(f"Loaded config: {filename}")
    print(f"Model description: {config['description']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    set_global_seed(constants["seed"])

    if device == "cuda":
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True


    # Dataset
    print("Loading dataset")
    dataset = GraphSkeletonDataset(
        root_dir=config["train_data_dir"],
        context_len_pre=config["context_len_pre"],
        context_len_post=config["context_len_post"],
        target_len=config["target_len"],
        step=config["step"]
    )
    print(f"Dataset ready with {len(dataset)} samples")

    torch_generator = torch.Generator()
    torch_generator.manual_seed(constants["seed"])
    
    n_total = len(dataset)
    n_val = max(1, int(n_total * config["validation_split"]))
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch_generator)

    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config["num_workers"], 
        pin_memory=True, 
        persistent_workers=True,
        generator=torch_generator,
        worker_init_fn=set_worker_seed
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=True, 
        persistent_workers=True,
        generator=torch_generator,
        worker_init_fn=set_worker_seed
    )


    # Model
    model = SkeletalMotionInterpolator(
        context_len_pre=config["context_len_pre"],
        context_len_post=config["context_len_post"],
        target_len=config["target_len"],
        rot_gnn_params=config["rot_gnn_params"],
        root_pos_mlp_params=config["root_pos_mlp_params"]
    )
    model = model.to(device)

    os.makedirs(constants["models_path"], exist_ok=True)
    model_path = constants["models_path"] + filename + constants["models_suffix"]


    # Logging
    os.makedirs(constants["train_log_path"], exist_ok=True)
    train_log_path = constants["train_log_path"] + filename + constants["log_suffix"]
    if os.path.exists(train_log_path):
        os.remove(train_log_path)

    log_str = lambda text: log_string(text=text, log_path=train_log_path)

    os.makedirs(constants["train_plot_path"], exist_ok=True)
    train_plot_path = constants["train_plot_path"] + filename + constants["plot_suffix"]
    if os.path.exists(train_plot_path):
        os.remove(train_plot_path)
    

    # Training preparation
    scheduler_params = config["lr_scheduler_params"]
    optimizer = torch.optim.Adam(model.parameters(), lr=scheduler_params["start_lr"])
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=scheduler_params["factor"],     
        patience=scheduler_params["patience"],     
        min_lr=scheduler_params["min_lr"]
    )

    l1_func = torch.nn.L1Loss()
    l2_func = torch.nn.MSELoss()

    parent_indices = dataset.parent_indices.to(device)
    offsets = dataset.offsets.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config["patience"]
    epochs = config["epochs"]

    train_losses = []
    val_losses = []
    val_rot_losses = []
    val_root_pos_losses = []
    val_fk_losses = []

    start_time = time.time()
    print("Starting time measurement")

    # Training 
    try:
        for epoch in range(1, epochs + 1):
            log_str(f"\n--- Epoch {epoch}/{epochs} ---")

            current_lr = optimizer.param_groups[0]["lr"]
            log_str(f"Learning rate:                  {current_lr:.3e}")
        
            model.train()

            total_loss = 0.0
            total_rot_loss = 0.0
            total_root_pos_loss = 0.0
            total_fk_loss = 0.0

            for batch in tqdm(train_loader, desc="Train", leave=False):
                
                batch = batch.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda', dtype=torch.bfloat16): 
                    out = model(batch)

                out["rot"] = out["rot"].float()
                out["root_pos"] = out["root_pos"].float()

                loss, rot_geo_loss, root_pos_loss, fk_pos_loss = calculate_loss(
                    out=out, 
                    batch=batch, 
                    l1_func=l1_func, 
                    l2_func=l2_func, 
                    offsets=offsets, 
                    parent_indices=parent_indices, 
                    loss_weights=config["loss_weights"]
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch.num_graphs
                total_rot_loss += rot_geo_loss.item() * batch.num_graphs
                total_root_pos_loss += root_pos_loss.item() * batch.num_graphs
                total_fk_loss += fk_pos_loss.item() * batch.num_graphs

            n_samples = len(train_dataset)
            avg_loss = total_loss / n_samples
            avg_rot_loss = total_rot_loss / n_samples
            avg_root_pos_loss = total_root_pos_loss / n_samples
            avg_fk_loss = total_fk_loss / n_samples

            train_losses.append(avg_loss)

            log_str(f"Training loss:                  {avg_loss:.7f}")
            log_str(f"'- Rotations geodesic L1:       '- {avg_rot_loss:.4f}")
            log_str(f"'- Root positions L2:           '- {avg_root_pos_loss:.4f}")
            log_str(f"'- FK positions L1:             '- {avg_fk_loss:.4f}")


            # Validation
            model.eval()

            total_loss = 0.0
            total_rot_loss = 0.0
            total_root_pos_loss = 0.0
            total_fk_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Val", leave=False):
                    
                    batch = batch.to(device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out = model(batch)

                    out["rot"] = out["rot"].float()
                    out["root_pos"] = out["root_pos"].float()

                    loss, rot_geo_loss, root_pos_loss, fk_pos_loss = calculate_loss(
                        out=out, 
                        batch=batch, 
                        l1_func=l1_func, 
                        l2_func=l2_func, 
                        offsets=offsets, 
                        parent_indices=parent_indices, 
                        loss_weights=config["loss_weights"]
                    )

                    total_loss += loss.item() * batch.num_graphs
                    total_rot_loss += rot_geo_loss.item() * batch.num_graphs
                    total_root_pos_loss += root_pos_loss.item() * batch.num_graphs
                    total_fk_loss += fk_pos_loss.item() * batch.num_graphs

            n_samples = len(val_dataset)
            avg_loss = total_loss / n_samples
            avg_rot_loss = total_rot_loss / n_samples
            avg_root_pos_loss = total_root_pos_loss / n_samples
            avg_fk_loss = total_fk_loss / n_samples

            val_losses.append(avg_loss)
            val_rot_losses.append(avg_rot_loss)
            val_root_pos_losses.append(avg_root_pos_loss)
            val_fk_losses.append(avg_fk_loss)

            log_str(f"Validation loss:                {avg_loss:.7f}")
            log_str(f"'- Rotations geodesic L1:       '- {avg_rot_loss:.4f}")
            log_str(f"'- Root positions L2:           '- {avg_root_pos_loss:.4f}")
            log_str(f"'- FK positions L1:             '- {avg_fk_loss:.4f}")

            scheduler.step(avg_loss)

            if avg_loss < (best_val_loss - config["early_stop_eps"]):
                best_val_loss = avg_loss
                epochs_no_improve = 0
                log_str("Validation loss improved, saving checkpoint")
                torch.save(model.state_dict(), model_path)
                log_str(f"Model saved to: {model_path}")
            else:
                epochs_no_improve += 1
                log_str(f"No improvement in validation loss for {epochs_no_improve} epochs")
                if epochs_no_improve >= patience:
                    log_str(f"Early stopped at epoch {epoch}")
                    print()
                    break

    except KeyboardInterrupt:
        log_str("\nTraining interrupted by user")

    finally:
        if epoch == epochs: print()

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        if len(val_losses) > 0:
            plt.figure(figsize=(12, 7))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train loss', linewidth=2)
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val loss', linewidth=2)

            plt.plot(range(1, len(val_rot_losses) + 1), val_rot_losses, label='Val rotations geodesic L1 loss', alpha=0.5)
            plt.plot(range(1, len(val_root_pos_losses) + 1), val_root_pos_losses, label='Val root positions L2 loss', alpha=0.5)
            plt.plot(range(1, len(val_fk_losses) + 1), val_fk_losses, label='Val FK positions L1 loss', alpha=0.5)
            
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.yscale('log')
            ax = plt.gca() 
            formatter = FormatStrFormatter('%g') 
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.title('Training and validation loss')
            plt.legend()
            plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7) 
            plt.tight_layout()
            plt.savefig(train_plot_path)
            plt.close()
        
        log_str(f"Total training time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
        log_str("Training complete")
        print("Cleaning memory and worker processes (this might take a few seconds)")

        del train_loader
        del val_loader
        gc.collect()

        print("Done")
        print()
