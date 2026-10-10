import argparse
from policies.reasoning_vla_policy import ReasoningVLAPolicy
from utils.websocket_server_policy import WebsocketPolicyServer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_option():
    parser = argparse.ArgumentParser()# /workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/reasoning_vla-3b-20251013-libero-sft/step-1710
    parser.add_argument('--env_type', type=str, default="libero", help="type of environment")# /root/reasoningVLA/outputs/ckpts/reasoning_vla-3b-20251012-libero-sft
    parser.add_argument('--ckpt_dir', type=str, default="/workspace/zhaochen/reasoningVLA-ruc/outputs/ckpts/BaM-InternVL-2b-20251204_2/step-69000", help="checkpoint directory")
    parser.add_argument('--window_size', type=int, default=1, help="window size")
    parser.add_argument('--backbone', type=str, default="internvl3.5", help="backbone model: qwen2.5vl, internvl3.5")
    parser.add_argument('--remain_ep_start', action='store_true', help="whether to remain episode start")
    parser.add_argument('--num_denoised_steps', type=int, default=5, help="number of denoised steps")
    parser.add_argument('--max_pad_state_and_action_length', type=int, default=64, help="max padding length")
    parser.add_argument('--port', type=int, default=8000, help="server port")
    
    args = parser.parse_args()
    return args

def deploy():
    opt = parse_option()
    # settings
    kwargs = {
        "env_type": opt.env_type,
        "ckpt_dir": opt.ckpt_dir,
        "window_size": opt.window_size,
        "remain_ep_start": opt.remain_ep_start,
        "num_denoised_steps": opt.num_denoised_steps,
        "max_pad_state_and_action_length": opt.max_pad_state_and_action_length,
        "backbone": opt.backbone, #qwen2.5vl, internvl3.5
        "device": "cuda:0"
    }
    policy = ReasoningVLAPolicy(**kwargs)
    
    print("Start serving.")
    host = "0.0.0.0"
    port = opt.port
    server = WebsocketPolicyServer(policy, host, port)
    server.serve_forever()

if __name__ == "__main__":
    deploy()