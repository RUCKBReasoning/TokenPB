import os
import torch
import argparse
import math
import time
import json

from utils.load_training_dataset import build_concat_streaming_dataset, create_dataloader_for_concat
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from model import FlowmatchingActionHeadConfig
from model import ReasoningVLAModel
from accelerate import Accelerator
from torch.optim import AdamW
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

'''
TODO: We can not use Zero-3 due to some unknown problems. See the same issue:
https://github.com/huggingface/accelerate/issues/3399
'''

def parse_option():
    parser = argparse.ArgumentParser()

    # args for the path of VLM, action expert, and FAST tokenizer
    parser.add_argument('--vlm_name_or_path', type = str, help="file path of pretrained VLM")
    parser.add_argument('--action_expert_name_or_path', type = str, 
                        help="path of the pretrained action expert. \
                            None (i.e., unset) means we will use a randomly initialized action expert.")
    parser.add_argument('--FAST_tokenizer_path', type = str, help="file path of pretrained FAST action tokenizer")
    
    # global args
    parser.add_argument('--per_device_train_batch_size', type = int, default = 8,
                        help = 'batch size per gpu device.')
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--epochs', type = int, default = 16)
    parser.add_argument('--save_ckpt_interval', type = int, default = 1)
    parser.add_argument('--save_step_interval', type = int, default = 2000)
    parser.add_argument('--lr', type = float, default = 1e-4, help = "learning rate")
    parser.add_argument('--tensorboard_log_dir', type = str, default = "./outputs/train_logs/optimus-3b-20250421")
    parser.add_argument('--output_ckpt_dir', type = str, default = "./outputs/ckpts/optimus-3b-20250421")

    # args for model training
    parser.add_argument('--tune_visual', action = 'store_true', help = "Whether to fine-tune the vision encoder part in the VLM")
    parser.add_argument('--tune_llm', action = 'store_true', help = "Whether to fine-tune the LLM part in the VLM")
    parser.add_argument('--tune_projector', action = 'store_true', help = "Whether to fine-tune the projectors in the action expert")
    parser.add_argument('--tune_dit', action = 'store_true', help = "Whether to fine-tune the diffusion transformers in the action expert")
    parser.add_argument('--detach_vlm_outputs_for_action_expert', action='store_true', 
                        help='whether to stop gradient from the action expert to VLM')
    
    # args for dataset
    parser.add_argument(
        '--dataset_path_list', type=str, nargs='+',
        help="List of training dataset paths, e.g. path1 path2 path3"
    )
    parser.add_argument('--mask_prompt_loss', action='store_true', help='whether to mask prompt loss')
    parser.add_argument('--add_discrete_action_tokens', action='store_true', help='whether to add discrete action tokens in VLM\'s output')
    parser.add_argument('--window_size', type = int, default = 1, help="size of the sliding window")
    parser.add_argument('--remain_ep_start_observations', action='store_true', help='whether to remain the episode start observations')
    parser.add_argument('--action_horizon', type = int, default = 32, help="size of the action chunk")
    parser.add_argument('--max_pad_state_and_action_length', type = int, default=64, help="dim num of the max padded state and action")
    parser.add_argument('--desc_type', type = str, default="qwen", help="dim num of the max padded state and action")
    parser.add_argument("--backbone_type", type = str, default="qwen2.5vl", help="qwen2.5vl or internvl3.5.")

    opt = parser.parse_args()

    return opt

def get_absolute_path(path):
    if os.path.isabs(path):
        return path
    else:
        current_path = os.getcwd()
        return os.path.join(current_path, path)

def print_model_parameters(model):
    for name, p in model.named_parameters():
        print(f"{name}: {p.numel()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"# all params: {total_params}")
    print(f"# trainable params: {trainable_params}")

def integrity_check(batch_data, processor):
    for input_id, label, attn_mask in zip(
        batch_data["input_ids"][0], batch_data["labels"][0], batch_data["attention_mask"][0]
    ):
        print(f"{input_id} -> {repr(processor.tokenizer.decode(input_id))} -> {label} -> {attn_mask}")
    print("integrity check ends.")

def train(opt):
    set_seed(opt.seed)

    writer = SummaryWriter(opt.tensorboard_log_dir)
    accelerator = Accelerator()
    accelerator.print(opt)

    total_batch_size = opt.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    accelerator.print(opt)
    accelerator.print("data items per batch:", total_batch_size)

    # # load dataset
    # dataset = ReasoningVLADataset(
    #     dataset_path_list = opt.dataset_path_list,
    #     mask_prompt_loss = opt.mask_prompt_loss,
    #     dataset_type = "for_vla"
    # )
    # accelerator.print("len(dataset):", len(dataset))

    # # dataloader
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     num_workers=16,
    #     prefetch_factor=4,
    #     batch_size=opt.per_device_train_batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     collate_fn=custom_collate_fn
    # )

    concat_dataset = build_concat_streaming_dataset(
        lerobot_data_paths=opt.dataset_path_list,
        model_name_or_path=opt.vlm_name_or_path,
        fast_tokenizer_path=opt.FAST_tokenizer_path,
        window_size=opt.window_size,
        action_horizon=opt.action_horizon,
        remain_ep_start_observations=opt.remain_ep_start_observations,
        process_mode="train",
        add_discrete_action_tokens=opt.add_discrete_action_tokens,
        mask_prompt_loss=opt.mask_prompt_loss,
        max_pad_state_and_action_length=opt.max_pad_state_and_action_length,
        desc_type=opt.desc_type,
        backbone = opt.backbone_type,
    )
    accelerator.print("len(concat_dataset):", len(concat_dataset))

    dataloader = create_dataloader_for_concat(
        concat_dataset,
        batch_size_per_device=opt.per_device_train_batch_size,
        num_workers=16,
        prefetch_factor=4
    )
    accelerator.print("len(dataloader):", len(dataloader))

    accelerator.print(f"using VLM from {opt.vlm_name_or_path}.")

    if opt.action_expert_name_or_path:
        accelerator.print(f"using a pre-trained action expert from {opt.action_expert_name_or_path}, thus we will load its configurations.")
        # load action expert's configurations
        action_expert_config_json = json.load(open(os.path.join(opt.action_expert_name_or_path, "action_expert_config.json")))
        action_expert_config = FlowmatchingActionHeadConfig(**action_expert_config_json)
    else:
        accelerator.print("using a randomly initialized action expert.")
        # manually set some important configurations because the action expert is randomly initialized
        action_expert_config = FlowmatchingActionHeadConfig(
            action_dim = opt.max_pad_state_and_action_length, # 64 is the max padding length of action and state vectors
            state_dim = opt.max_pad_state_and_action_length,
            action_horizon = opt.action_horizon
        )

    # initialize ReasoningVLA model
    model = ReasoningVLAModel(
        vlm_name_or_path = opt.vlm_name_or_path,
        action_expert_name_or_path = opt.action_expert_name_or_path,
        action_expert_config = action_expert_config,
        tune_llm = opt.tune_llm,
        tune_visual = opt.tune_visual,
        tune_projector = opt.tune_projector,
        tune_dit = opt.tune_dit,
        detach_vlm_outputs_for_action_expert = opt.detach_vlm_outputs_for_action_expert
    )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters in model: {total_params}")

        total_backbone_params = sum(p.numel() for p in model.backbone.parameters())
        print(f"Total parameters in model.backbone: {total_backbone_params}")

        total_action_expert_params = sum(p.numel() for p in model.action_expert.parameters())
        print(f"Total parameters in model.action_expert: {total_action_expert_params}")

        total_dit_params = sum(p.numel() for p in model.action_expert.model.parameters())
        print(f"Total parameters in model.action_expert.DiT: {total_dit_params}")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params

        print(f"Trainable parameters: {trainable_params}")
        print(f"Trainable ratio: {trainable_ratio:.2%}")

    # load AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr = opt.lr, betas = (0.95, 0.999),
        eps = 1e-8, weight_decay = 1e-5
    )

    # learning rate scheduler (linear warm up and cosine decay)
    num_total_batches = math.ceil(opt.epochs * math.ceil(len(concat_dataset) / total_batch_size))
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer = optimizer,
        warmup_epochs = int(num_total_batches * 0.02) * accelerator.num_processes,
        max_epochs = num_total_batches * accelerator.num_processes,
        warmup_start_lr = 0.0,
        eta_min = 0.1 * opt.lr
    )

    model = model.to(torch.bfloat16)
    optimizer, model, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, model, dataloader, lr_scheduler
    )
    if accelerator.is_main_process:
        _m = accelerator.unwrap_model(model)
        print("DEBUG dtype after prepare -> VLM:", next(_m.backbone.model.parameters()).dtype,
              "action_expert:", next(_m.action_expert.parameters()).dtype, flush=True)

    global_completed_steps = 0
    accumulation_loss = 0
    action_loss = 0
    pretrain_loss = 0
    # set model to the train() mode
    model.train()
        
    st = time.time()
    for epoch in range(opt.epochs):
        # set the epoch into each dataset
        for ds in concat_dataset.datasets:
            ds.set_epoch(epoch)

        accelerator.print(f"{epoch=}")
        for batch_idx, batch in enumerate(dataloader):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and batch_idx==0 and epoch==0:
                integrity_check(batch, model.backbone.processor)
            # accelerator.print("batch_idx:", batch_idx)
            with accelerator.accumulate(model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
                    loss = outputs.loss

                accumulation_loss += loss.detach().float()
                action_loss += outputs.action_expert_loss.detach().float()
                pretrain_loss += outputs.vlm_loss.detach().float()
                # when deepspeed is enabled, `accelerator.backward(loss)` is doing optimizer.step(), optimizer.zero_grad(), and grad accumulation automatically. 
                # see `if self.is_gradient_accumulation_boundary():` line in path-to-env/site-packages/deepspeed/runtime/engine.py
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 'accelerator.sync_gradients' checks if the accelerator has performed an optimization step on the `total_batch_size` data samples
            if accelerator.sync_gradients:
                global_completed_steps += 1
                accelerator.print("GPU 0, step {}, loss {}".format(global_completed_steps, accumulation_loss / accelerator.gradient_accumulation_steps))
                _ggn = getattr(model, "get_global_grad_norm", None)  # deepspeed-only; skip if absent
                accelerator.print("GPU 0, step {}, grad_norm {}".format(global_completed_steps, _ggn() if callable(_ggn) else None))
                accelerator.print("GPU 0, step {}, action_loss {}, pretrain_loss {}".format(global_completed_steps, action_loss / accelerator.gradient_accumulation_steps, pretrain_loss / accelerator.gradient_accumulation_steps))
                accelerator.print("GPU 0, step {}, lr state dict:".format(global_completed_steps), lr_scheduler.state_dict())
                accelerator.print(f"A batch spends {time.time()-st} seconds")
                
                writer.add_scalar(
                    'train-loss/gpu-{}'.format(accelerator.process_index), 
                    accumulation_loss / accelerator.gradient_accumulation_steps, 
                    global_completed_steps
                )
                writer.add_scalar(
                    'learning-rate/gpu-{}'.format(accelerator.process_index), 
                    lr_scheduler.get_last_lr()[0], 
                    global_completed_steps
                )
                # reset accumulation_loss to 0
                accumulation_loss = 0
                action_loss = 0
                pretrain_loss = 0
                # reset start time
                st = time.time()
            
                if global_completed_steps % opt.save_step_interval == 0 and global_completed_steps != 0:
                    accelerator.print("save model")
                    # save checkpoint
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(os.path.join(opt.output_ckpt_dir, f"step-{global_completed_steps}"))

        if epoch % opt.save_ckpt_interval == 0 or epoch == opt.epochs-1:
            accelerator.print("save model")
            # save checkpoint
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(opt.output_ckpt_dir, f"step-{global_completed_steps}"))

if __name__ == "__main__":
    opt = parse_option()
    train(opt)