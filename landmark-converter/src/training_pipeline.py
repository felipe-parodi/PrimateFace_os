"""Training pipeline and utilities for landmark converter models.

This module provides the ModelTrainer class which handles the complete
training loop including validation, visualization, and checkpointing.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import json

from utils.train_utils import visualize_predictions, plot_training_curve, plot_mpjpe_per_keypoint_bar, plot_mpjpe_heatmap_on_sample, save_test_evaluation_results
from utils.data_utils import normalize_keypoints_bbox, denormalize_keypoints_bbox

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device, output_dir, args,
                 train_loader, val_loader, test_loader=None,
                 num_source_kpt=None, target_kpt_slice_idx=None,
                 model_expects_graph_data=False):
        """
        Initializes the ModelTrainer.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (torch.nn.Module): The loss function.
            device (torch.device): The device to train on (e.g., 'cuda', 'cpu').
            output_dir (str): Directory to save checkpoints, logs, and plots.
            args (argparse.Namespace): Parsed command-line arguments containing training hyperparameters
                                       (epochs, lr, batch_size, vis_every_n_epochs, etc.)
                                       and model-specific configurations if needed for saving.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader, optional): DataLoader for the test set. Defaults to None.
            num_source_kpt (int, optional): Number of source keypoints. Used by training/evaluation logic.
            target_kpt_slice_idx (int, optional): Index to slice target keypoints. Typically 0 or 1.
            model_expects_graph_data (bool): If True, handles PyG Batch objects.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.num_source_kpt = num_source_kpt if num_source_kpt is not None else args.num_source_kpt
        self.target_kpt_slice_idx = target_kpt_slice_idx if target_kpt_slice_idx is not None else getattr(args, 'target_kpt_slice_idx', 1)
        
        self.model_expects_graph_data = model_expects_graph_data

        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(self.output_dir, "best_model.pth")
        
        self.epochs_log = []
        self.train_losses_log = []
        self.val_losses_log = []
        
        self.start_epoch = 1
        self.parsed_overlap_mapping = None # Initialize

        # Priority: 1. --overlap_mapping string, 2. --use_fixed_mapping_68_to_49
        custom_mapping_str = getattr(self.args, 'overlap_mapping', None)
        use_fixed_map_flag = getattr(self.args, 'use_fixed_mapping_68_to_49', False)

        if custom_mapping_str:
            self.parsed_overlap_mapping = self._parse_overlap_mapping(custom_mapping_str)
            if self.parsed_overlap_mapping and use_fixed_map_flag:
                print("Info: Both --overlap_mapping string and --use_fixed_mapping_68_to_49 are set. Using the string version.")
        elif use_fixed_map_flag:
            # Predefined mapping from 68-point source to 49-point target (full, before slicing)
            # {source_idx_68: target_full_idx_49}
            fixed_map_68_to_49_full = {
                48: 6, 8: 7, 54: 8, 27: 13, 39: 14, 36: 16, 42: 19, 45: 21,
                32: 32, 34: 33, 30: 35, 51: 36, 57: 37, 28: 39, 29: 40,
                49: 41, 52: 43, 53: 44, 55: 45, 56: 46, 58: 47, 59: 48
            }
            
            current_source_kpts = self.num_source_kpt # This is args.num_source_kpt
            current_total_target_kpts_in_data = self.args.num_target_kpt

            if current_source_kpts == 68 and current_total_target_kpts_in_data == 49:
                temp_mapping = []
                num_kpts_model_predicts = current_total_target_kpts_in_data - self.target_kpt_slice_idx
                for src_idx_68, target_full_idx_49 in fixed_map_68_to_49_full.items():
                    target_pred_idx = target_full_idx_49 - self.target_kpt_slice_idx
                    
                    # Validate source index against actual source keypoints used by model
                    if not (0 <= src_idx_68 < current_source_kpts):
                        # This should ideally not happen if map is correct and current_source_kpts is 68
                        # print(f"Warning (fixed_map): Source index {src_idx_68} from fixed map is out of bounds for model's source keypoints ({current_source_kpts}). Skipping this pair.")
                        continue

                    # Validate target prediction index against what model actually predicts
                    if 0 <= target_pred_idx < num_kpts_model_predicts:
                        temp_mapping.append([src_idx_68, target_pred_idx])
                    # else: The keypoint was part of the slice, so we don't include it in mapping for predicted keypoints
                
                if temp_mapping:
                    self.parsed_overlap_mapping = temp_mapping
                    print(f"Info: Using fixed 68->49 overlap mapping, {len(temp_mapping)} pairs configured. Effective if --enforce_overlap is also set.")
                else:
                    print("Warning: --use_fixed_mapping_68_to_49 is set, but the fixed mapping resulted in no valid overlaps for the current target_kpt_slice_idx. No fixed mapping will be applied.")
            else:
                print("Warning: --use_fixed_mapping_68_to_49 is set, but current model configuration (num_source_kpt, num_target_kpt) is not 68 and 49 respectively. Fixed mapping not applied.")

        # Updated informational messages
        if self.parsed_overlap_mapping and self.args.enforce_overlap:
            enforce_duration = self.args.enforce_overlap_epochs
            if enforce_duration > 0:
                duration_str = f"for the first {enforce_duration} epoch(s)" if enforce_duration > 0 else "(duration is 0, effectively no enforcement)"
                print(f"Info: Overlap mapping is configured and --enforce_overlap is set. Mapping will be applied as initialization {duration_str}.")
            else:
                print("Info: Overlap mapping is configured and --enforce_overlap is set, but --enforce_overlap_epochs is 0. Mapping will NOT be applied during training epochs.")
        elif self.parsed_overlap_mapping and not self.args.enforce_overlap:
            print("Info: Overlap mapping is configured, but --enforce_overlap is NOT set. Mapping will NOT be applied.")
        elif not self.parsed_overlap_mapping and self.args.enforce_overlap:
            print("Warning: --enforce_overlap is set, but no valid overlap mapping (neither custom string nor fixed) is configured. No overlap will be enforced.")

        os.makedirs(self.output_dir, exist_ok=True)
        if self.model_expects_graph_data:
            print("ModelTrainer configured to expect PyTorch Geometric Batch objects.")

    def _train_epoch(self, current_epoch):
        self.model.train()
        total_loss = 0

        if self.model_expects_graph_data:
            for batch_idx, data_batch in enumerate(self.train_loader):
                data_batch = data_batch.to(self.device)
                
                self.optimizer.zero_grad()
                predicted_kpts_structured = self.model(data_batch)
                
                target_kpts_for_loss = data_batch.y.view(-1, self.model.num_target_kpts, 2)
                
                loss = self.criterion(predicted_kpts_structured, target_kpts_for_loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        else:
            for batch_idx, (img_tensors, source_kpts_raw, target_kpts_raw, bboxes, _) in enumerate(self.train_loader):
                source_kpts_raw = source_kpts_raw.to(self.device).float()
                target_kpts_raw = target_kpts_raw.to(self.device).float()
                bboxes = bboxes.to(self.device).float()
                batch_size = source_kpts_raw.size(0)

                source_kpts_xy_raw = source_kpts_raw[..., :2]
                norm_source_kpts_xy = normalize_keypoints_bbox(source_kpts_xy_raw, bboxes)
                input_kpts_flat = norm_source_kpts_xy.contiguous().view(batch_size, -1)

                self.optimizer.zero_grad()
                predicted_kpts_norm_structured = self.model(input_kpts_flat)
                
                # --- Enforce Overlap Logic (Configurable Duration) ---
                if self.args.enforce_overlap and self.parsed_overlap_mapping and current_epoch <= self.args.enforce_overlap_epochs:
                    if current_epoch == 1 and self.args.enforce_overlap_epochs > 0 : # Print only on first effective epoch of enforcement
                        print(f"Info: Applying overlap enforcement for epoch {current_epoch} (up to {self.args.enforce_overlap_epochs} epochs)...")
                    for src_idx, target_pred_idx in self.parsed_overlap_mapping:
                        if not (0 <= src_idx < norm_source_kpts_xy.shape[1] and \
                                0 <= target_pred_idx < predicted_kpts_norm_structured.shape[1]):
                            print(f"Warning (Train Epoch {current_epoch} Init): Overlap mapping [src:{src_idx}, pred_tgt:{target_pred_idx}] out of bounds. Skipping this pair.")
                            continue
                        predicted_kpts_norm_structured[:, target_pred_idx, :] = norm_source_kpts_xy[:, src_idx, :]
                # --- End Enforce Overlap Logic ---

                target_kpts_raw_sliced = target_kpts_raw[:, self.target_kpt_slice_idx:, :]
                target_kpts_xy_raw_sliced = target_kpts_raw_sliced[..., :2]
                norm_target_kpts_xy_sliced = normalize_keypoints_bbox(target_kpts_xy_raw_sliced, bboxes)
                
                loss = self.criterion(predicted_kpts_norm_structured, norm_target_kpts_xy_sliced)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _evaluate_model(self, data_loader, calc_mpjpe=True):
        self.model.eval()
        total_loss = 0
        all_denorm_predicted_kpts_list = [] 
        all_raw_target_kpts_sliced_list = []
        
        with torch.no_grad():
            if self.model_expects_graph_data:
                for batch_idx, data_batch in enumerate(data_loader):
                    data_batch = data_batch.to(self.device)
                    batch_size_gnn = data_batch.num_graphs

                    predicted_kpts_structured = self.model(data_batch)
                    
                    target_kpts_for_loss = data_batch.y.view(-1, self.model.num_target_kpts, 2)
                    loss = self.criterion(predicted_kpts_structured, target_kpts_for_loss)
                    total_loss += loss.item()

                    if calc_mpjpe:
                        # Denormalize all predictions in the batch first
                        current_batch_bboxes = data_batch.bbox.view(batch_size_gnn, 4)
                        denorm_predicted_kpts_batch = denormalize_keypoints_bbox(predicted_kpts_structured, current_batch_bboxes) # [B, K_pred, 2]
                        
                        # Get individual Data objects from the PyG Batch
                        data_list = data_batch.to_data_list()

                        for i in range(batch_size_gnn):
                            current_data_sample = data_list[i]
                            
                            # Access raw target keypoints from the individual Data object
                            if hasattr(current_data_sample, 'target_kpts_with_vis') and isinstance(current_data_sample.target_kpts_with_vis, torch.Tensor):
                                original_target_kpts_sample_full = current_data_sample.target_kpts_with_vis.to(self.device) # [K_trg_total, 3]
                            else:
                                print(f"Warning: target_kpts_with_vis not found or not a tensor in Data sample {i}. Skipping MPJPE for this item.")
                                continue # Skip this item if data is missing

                            target_kpts_sample_raw_sliced_xy = original_target_kpts_sample_full[self.target_kpt_slice_idx:, :2] # [K_trg_pred, 2]

                            # Append the i-th prediction and its corresponding target
                            all_denorm_predicted_kpts_list.append(denorm_predicted_kpts_batch[i].cpu().numpy())
                            all_raw_target_kpts_sliced_list.append(target_kpts_sample_raw_sliced_xy.cpu().numpy())

            else:
                for img_tensors, source_kpts_raw, target_kpts_raw, bboxes, img_paths in data_loader:
                    source_kpts_raw = source_kpts_raw.to(self.device).float()
                    target_kpts_raw = target_kpts_raw.to(self.device).float()
                    bboxes = bboxes.to(self.device).float()
                    batch_size = source_kpts_raw.size(0)

                    source_kpts_xy_raw = source_kpts_raw[..., :2]
                    norm_source_kpts_xy = normalize_keypoints_bbox(source_kpts_xy_raw, bboxes)
                    input_kpts_flat = norm_source_kpts_xy.contiguous().view(batch_size, -1)

                    predicted_kpts_norm_structured = self.model(input_kpts_flat)
                    
                    # --- REMOVED Enforce Overlap Logic (Evaluation) ---
                    # Evaluation should use raw model output if training is epoch 1 init.
                    # --- End REMOVED Enforce Overlap Logic ---
                    
                    target_kpts_raw_sliced = target_kpts_raw[:, self.target_kpt_slice_idx:, :]
                    target_kpts_xy_raw_sliced = target_kpts_raw_sliced[..., :2]
                    norm_target_kpts_xy_sliced = normalize_keypoints_bbox(target_kpts_xy_raw_sliced, bboxes)
                    
                    loss = self.criterion(predicted_kpts_norm_structured, norm_target_kpts_xy_sliced)
                    total_loss += loss.item()

                    if calc_mpjpe:
                        denormalized_preds_batch = denormalize_keypoints_bbox(predicted_kpts_norm_structured, bboxes)
                        
                        for i in range(batch_size):
                            all_denorm_predicted_kpts_list.append(denormalized_preds_batch[i].cpu().numpy())
                            all_raw_target_kpts_sliced_list.append(target_kpts_xy_raw_sliced[i].cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        mpjpe_overall = None
        mpjpe_per_keypoint = None

        if calc_mpjpe and len(all_denorm_predicted_kpts_list) > 0 and len(all_denorm_predicted_kpts_list) == len(all_raw_target_kpts_sliced_list):
            preds_pixel_np = np.array(all_denorm_predicted_kpts_list) 
            targets_pixel_np = np.array(all_raw_target_kpts_sliced_list)
            
            if preds_pixel_np.shape == targets_pixel_np.shape:
                distances = np.sqrt(np.sum((preds_pixel_np - targets_pixel_np)**2, axis=2))
                mpjpe_overall = np.mean(distances)
                mpjpe_per_keypoint = np.mean(distances, axis=0)
            else:
                print(f"MPJPE calculation warning: Shape mismatch preds ({preds_pixel_np.shape}) vs targets ({targets_pixel_np.shape}).")
        elif calc_mpjpe:
            print("MPJPE calculation warning: Not enough data or mismatch for MPJPE.")
            
        return avg_loss, mpjpe_overall, mpjpe_per_keypoint
        
    def _save_checkpoint(self, epoch, current_val_loss, is_best=False):
        """Saves model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': current_val_loss,
            'args': self.args, # Save training arguments
            'train_losses_log': self.train_losses_log,
            'val_losses_log': self.val_losses_log,
            'epochs_log': self.epochs_log,
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            torch.save(checkpoint_data, self.best_model_path)
            print(f"Best model (Epoch {epoch}) saved to {self.best_model_path} with Val Loss: {current_val_loss:.4f}")
        
        # Optional: save latest checkpoint
        # latest_model_path = os.path.join(self.output_dir, "latest_model.pth")
        # torch.save(checkpoint_data, latest_model_path)
        # print(f"Latest checkpoint (Epoch {epoch}) saved to {latest_model_path}")


    def _load_checkpoint(self, checkpoint_path):
        """Loads model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}. Training from scratch.")
            return False

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf'))) # Handle older checkpoints
        
        # Restore logs if available
        if 'train_losses_log' in checkpoint and 'val_losses_log' in checkpoint and 'epochs_log' in checkpoint:
            self.train_losses_log = checkpoint['train_losses_log']
            self.val_losses_log = checkpoint['val_losses_log']
            self.epochs_log = checkpoint['epochs_log']
            # Ensure logs are lists, not numpy arrays from older saves
            self.train_losses_log = list(self.train_losses_log)
            self.val_losses_log = list(self.val_losses_log)
            self.epochs_log = list(self.epochs_log)

        print(f"Resumed from epoch {checkpoint['epoch']}. Previous best_val_loss: {self.best_val_loss:.4f}")
        # Sync optimizer's device if model moved
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        return True

    def _parse_overlap_mapping(self, mapping_str):
        if not mapping_str:
            return None
        try:
            parsed_mapping = json.loads(mapping_str)
            if not isinstance(parsed_mapping, list) or not all(isinstance(item, list) and len(item) == 2 and all(isinstance(i, int) for i in item) for item in parsed_mapping):
                print("Warning: --overlap_mapping is not a list of [src_idx, target_pred_idx] pairs. Ignoring.")
                return None
            # Validate indices further if needed (e.g., against num_source_kpt and num_target_kpt_predicted)
            # For now, basic structure validation.
            return parsed_mapping
        except json.JSONDecodeError:
            print(f"Warning: Could not parse --overlap_mapping string: {mapping_str}. Expected JSON format like '[[0,0],[1,2]]'. Ignoring.")
            return None

    def train(self):
        """
        Main training loop.
        """
        print("Starting training process...")
        
        # Resume if checkpoint path is provided in args and exists
        if hasattr(self.args, 'resume_checkpoint') and self.args.resume_checkpoint:
            self._load_checkpoint(self.args.resume_checkpoint)
        elif os.path.exists(self.best_model_path) and getattr(self.args, 'resume_if_best_exists', False): # Optional: auto-resume if best exists
            print("Best model path exists, attempting to resume...")
            self._load_checkpoint(self.best_model_path)


        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print(f"--- Epoch {epoch}/{self.args.epochs} ---")
            
            train_loss = self._train_epoch(epoch)
            val_loss, val_mpjpe, _ = self._evaluate_model(self.val_loader, calc_mpjpe=True) 

            self.epochs_log.append(epoch)
            self.train_losses_log.append(train_loss)
            self.val_losses_log.append(val_loss)

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end="")
            if val_mpjpe is not None: print(f", Val MPJPE: {val_mpjpe:.2f} pixels", end="")
            print() # Newline

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
            # else: # Optionally save latest checkpoint even if not best
            #    self._save_checkpoint(epoch, val_loss, is_best=False)


            if epoch % self.args.vis_every_n_epochs == 0 or epoch == self.args.epochs:
                if self.val_loader and len(self.val_loader.dataset) > 0:
                    print(f"Visualizing predictions for epoch {epoch}...")
                    vis_subset_len = min(getattr(self.args, 'vis_num_samples', 3), len(self.val_loader.dataset))
                    
                    current_val_dataset = self.val_loader.dataset
                    if vis_subset_len > len(current_val_dataset): 
                        vis_subset_len = len(current_val_dataset)

                    if vis_subset_len > 0:
                        vis_indices = random.sample(range(len(current_val_dataset)), vis_subset_len)
                        vis_val_subset = torch.utils.data.Subset(current_val_dataset, vis_indices)
                        
                        visualize_predictions(
                            model=self.model, 
                            dataset_subset=vis_val_subset, 
                            device=self.device,
                            num_samples=vis_subset_len, 
                            output_dir=self.output_dir, 
                            epoch_num=str(epoch),
                            save_pdf=self.args.save_plots_as_pdf,
                            target_kpt_slice_idx=self.target_kpt_slice_idx,
                            args_config=self.args,
                            model_expects_graph_data=self.model_expects_graph_data
                        )
                    else:
                        print("Validation set is too small or empty for visualization.")
                else:
                    print("No validation loader or empty validation set, skipping visualization.")

        print("Training finished.")
        self._log_training_progress()
        self._plot_training_curves()
        
        return self.best_model_path


    def _log_training_progress(self):
        log_df = pd.DataFrame({
            'epoch': self.epochs_log, 
            'train_loss': self.train_losses_log, 
            'val_loss': self.val_losses_log
        })
        log_csv_path = os.path.join(self.output_dir, "training_log.csv")
        try:
            log_df.to_csv(log_csv_path, index=False)
            print(f"Training log saved to {log_csv_path}")
        except Exception as e:
            print(f"Error saving training log: {e}")

    def _plot_training_curves(self):
        plot_training_curve(
            self.epochs_log, self.train_losses_log, self.val_losses_log, 
            self.output_dir, 
            base_filename="training_curve",
            save_pdf=self.args.save_plots_as_pdf
        )

    def evaluate_on_test_set(self, model_path_to_load=None):
        if not self.test_loader:
            print("No test loader provided. Skipping test set evaluation.")
            return None

        if model_path_to_load is None:
            model_path_to_load = self.best_model_path
        
        if not os.path.exists(model_path_to_load):
            print(f"Model file not found: {model_path_to_load}. Cannot evaluate on test set.")
            return None

        print(f"Loading model from {model_path_to_load} for test set evaluation...")
        checkpoint = torch.load(model_path_to_load, map_location=self.device)
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            return None


        print("Evaluating on Test Set...")
        test_loss, test_mpjpe, test_mpjpe_per_keypoint = self._evaluate_model(self.test_loader, calc_mpjpe=True)

        print(f"Test Set - Avg Loss: {test_loss:.4f}")
        if test_mpjpe is not None: print(f"Test Set - Avg MPJPE (pixels): {test_mpjpe:.2f}")

        test_metrics = {"avg_loss": test_loss, "avg_mpjpe_pixels": test_mpjpe}
        if test_mpjpe_per_keypoint is not None:
            print(f"Test Set - Avg MPJPE per keypoint (pixels): {test_mpjpe_per_keypoint.tolist()}")
            test_metrics["avg_mpjpe_per_keypoint_pixels"] = test_mpjpe_per_keypoint.tolist()
            
            num_kpts_for_bar_label = self.args.num_target_kpt - self.target_kpt_slice_idx
            plot_mpjpe_per_keypoint_bar(
                mpjpe_values=test_mpjpe_per_keypoint, 
                output_dir=self.output_dir,
                base_filename="test_mpjpe_per_keypoint",
                save_pdf=self.args.save_plots_as_pdf,
                num_target_kpts_bar_label=str(num_kpts_for_bar_label)
            )

            if self.test_loader and len(self.test_loader.dataset) > 0:
                current_test_dataset = self.test_loader.dataset
                sample_idx = random.randint(0, len(current_test_dataset) - 1)
                
                if self.model_expects_graph_data:
                    if isinstance(current_test_dataset, torch.utils.data.Subset):
                        data_sample_pyg = current_test_dataset.dataset[current_test_dataset.indices[sample_idx]]
                    else:
                        data_sample_pyg = current_test_dataset[sample_idx]
                    
                    img_path_sample = data_sample_pyg.image_path
                    bbox_sample = data_sample_pyg.bbox
                    original_target_kpts_for_heatmap = data_sample_pyg.target_kpts_with_vis 
                else:
                    _, _, target_kpts_raw_sample, bbox_sample_tuple, img_path_sample = current_test_dataset[sample_idx]
                    original_target_kpts_for_heatmap = target_kpts_raw_sample
                    bbox_sample = bbox_sample_tuple

                effective_target_kpts_sample_for_heatmap_xy = original_target_kpts_for_heatmap[self.target_kpt_slice_idx:, :2]
                
                try:
                    target_kpts_denorm_for_heatmap = effective_target_kpts_sample_for_heatmap_xy.cpu().numpy()

                    plot_mpjpe_heatmap_on_sample(
                        image_path_sample=img_path_sample,
                        target_keypoints_original_scale=target_kpts_denorm_for_heatmap,
                        mpjpe_per_keypoint=test_mpjpe_per_keypoint,
                        output_dir=self.output_dir,
                        base_filename="test_mpjpe_heatmap_sample",
                        save_pdf=self.args.save_plots_as_pdf
                    )
                except Exception as e:
                    print(f"Error during heatmap plotting: {e}")

        model_config = {
            "model_class": self.model.__class__.__name__,
            "num_source_kpts_data": self.num_source_kpt,
            "num_target_kpts_model_predicts": self.args.num_target_kpt - self.target_kpt_slice_idx,
        }
        if self.model_expects_graph_data and hasattr(self.args, 'gcn_hidden_channels'):
            model_config['gcn_hidden_channels'] = self.args.gcn_hidden_channels
            model_config['num_gcn_layers'] = self.args.num_gcn_layers
            model_config['node_input_features'] = self.model.node_input_features
        elif hasattr(self.model, 'hidden_dim1'): 
            model_config['hidden_dim1'] = self.model.hidden_dim1


        save_test_evaluation_results(
            test_metrics=test_metrics,
            model_config=model_config,
            training_args=self.args,
            output_dir=self.output_dir,
            base_filename=f"test_results_{self.model.__class__.__name__.lower()}" 
        )
        
        if self.test_loader and len(self.test_loader.dataset) > 0:
            print("Visualizing predictions on Test Set samples...")
            current_test_dataset = self.test_loader.dataset
            vis_subset_len_test = min(getattr(self.args, 'vis_num_samples_test', 3), len(current_test_dataset))

            if vis_subset_len_test > 0:
                vis_indices_test = random.sample(range(len(current_test_dataset)), vis_subset_len_test)
                vis_test_subset = torch.utils.data.Subset(current_test_dataset, vis_indices_test)
                
                visualize_predictions(
                    model=self.model, 
                    dataset_subset=vis_test_subset,
                    device=self.device,
                    num_samples=vis_subset_len_test, 
                    output_dir=self.output_dir, 
                    epoch_num="final_test", 
                    save_pdf=self.args.save_plots_as_pdf,
                    target_kpt_slice_idx=self.target_kpt_slice_idx,
                    args_config=self.args,
                    model_expects_graph_data=self.model_expects_graph_data
                )
            else:
                print("Test set too small or empty for visualization.")
        
        return test_metrics

# # Example usage (conceptual, would be in train_*.py scripts)
# if __name__ == '__main__':
    # This is a placeholder for how it might be used.
    # Actual usage will be in train_converter.py

    # 1. Parse arguments (epochs, lr, data paths, model type, etc.)
    # Mock args for demonstration
    class MockArgs:
        epochs = 2 # Small number for demo
        learning_rate = 1e-3
        batch_size = 4 # Small
        output_dir = "./outputs/trainer_test"
        coco_json = "path/to/your/coco.json" # Needs a real path for dataset
        source_field = 'keypoints_68'
        target_field = 'keypoints'
        source_num_field = 'num_keypoints_68'
        target_num_field = 'num_keypoints'
        num_source_kpt = 68
        num_target_kpt = 49 # Actual number in data, model predicts N-1
        
        # Model specific (e.g. for SimpleLinearConverter or KeypointConverterMLP)
        # hidden_dim1 = 128 
        # hidden_dim2 = 128

        vis_every_n_epochs = 1
        save_plots_as_pdf = False
        split_strategy = "random"
        train_ratio = 0.1 # Use small dataset for demo
        val_ratio = 0.05
        num_workers = 0
        split_random_seed = 42
        gpu_id = None # or 0, 1 etc.
        resume_checkpoint = None # or path to a checkpoint
        target_kpt_slice_idx = 1 # default is 1
        image_base_dir = None # Add missing attribute
        visualize_raw_data = False # Add missing attribute

    args = MockArgs()

    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")

    # 3. Initialize Dataset and DataLoaders
    # This requires CocoPairedKeypointDataset, transforms etc.
    # For now, assume train_loader, val_loader, test_loader are created.
    # from utils.data_utils import CocoPairedKeypointDataset
    # from torchvision import transforms
    # from torch.utils.data import DataLoader, random_split
    
    # Dummy Dataloaders for structure
    # In reality, these would load actual data.
    # Need `models.py` for SimpleLinearConverter
    try:
        from src.models import SimpleLinearConverter # Import from src.models
        from utils.data_utils import CocoPairedKeypointDataset, prepare_data # Assuming utils are accessible
        from torchvision import transforms

        # Create a dummy COCO JSON for testing if it doesn't exist
        dummy_coco_path = os.path.join(args.output_dir, "dummy_coco.json")
        os.makedirs(args.output_dir, exist_ok=True)
        if not os.path.exists(dummy_coco_path) or True: # always create for demo simplicity
            dummy_coco_data = {
                "images": [{"id": i, "file_name": f"dummy_{i}.jpg", "height": 256, "width": 256} for i in range(20)],
                "annotations": [
                    {
                        "id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                        "bbox": [random.randint(0,50), random.randint(0,50), random.randint(100,150), random.randint(100,150)],
                        "keypoints_68": [random.random()*256 for _ in range(68*3)], # x,y,v
                        "num_keypoints_68": 68,
                        "keypoints": [random.random()*256 for _ in range(49*3)], # x,y,v
                        "num_keypoints": 49,
                    } for i in range(20)
                ],
                "categories": [{"id": 1, "name": "face", "keypoints": ["kp"+str(j) for j in range(49)], "skeleton": []}]
            }
            import json
            with open(dummy_coco_path, 'w') as f:
                json.dump(dummy_coco_data, f)
        args.coco_json = dummy_coco_path


        img_transforms = transforms.Compose([transforms.ToTensor()]) 
        # Use prepare_data for consistency
        train_loader, val_loader, test_loader, _ = prepare_data(
            args, 
            img_transforms, 
            CocoPairedKeypointDataset
        )
        
        if train_loader and val_loader:
            # 4. Initialize Model
            model = SimpleLinearConverter(
                num_source_kpts=args.num_source_kpt,
                num_target_kpts=args.num_target_kpt - args.target_kpt_slice_idx 
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            criterion = torch.nn.MSELoss()

            trainer = ModelTrainer(
                model=model, optimizer=optimizer, criterion=criterion, device=device,
                output_dir=args.output_dir, args=args,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                num_source_kpt=args.num_source_kpt,
                target_kpt_slice_idx=args.target_kpt_slice_idx,
                model_expects_graph_data=False
            )

            print("Starting dummy training run...")
            trainer.train()
            print("Starting dummy test set evaluation...")
            trainer.evaluate_on_test_set()
            print(f"Dummy run completed. Check outputs in {args.output_dir}")
        else:
            print("Skipping trainer demo due to empty dataloaders (dataset likely too small or error in split).")

    except ImportError as e:
        print(f"Could not import necessary modules for demo: {e}. Skipping dummy run.")
    except Exception as e:
        print(f"An error occurred during the dummy run setup: {e}") 