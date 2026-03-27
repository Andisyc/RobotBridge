import time

import numpy as np
from loguru import logger

from agents.base_agent import BaseAgent
import torch

class GMTAgent(BaseAgent):
    def __init__(self, config, env):
        self.config = config
        self.time = 0
        self.print_cnt = 0
        self.env = env
        self.device = self.config.get('device', 'cpu')

        self.load_policy()

    def _inference(self, obs_buf_dict):
        obs_numpy = obs_buf_dict['obs'] 
        obs_tensor = torch.from_numpy(obs_numpy).float().to(self.device)
        
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # with torch.no_grad():
        #     action_tensor = self.policy(obs_tensor)
        
        with torch.no_grad():
            if hasattr(self.policy, 'run'):
                # === ONNX 推理模式 ===
                # 取出 numpy 格式的观测值喂给 ONNX
                obs_np = obs_tensor.cpu().numpy().astype(np.float32)
                # --- 终极保底：强制切片适配 770 维模型 ---
                if obs_np.shape[1] > 770:
                    obs_np = obs_np[:, :770]  # 只取前 770 维核心数据
                # ------------------------------------
                input_name = self.policy.get_inputs()[0].name
                action_np = self.policy.run(None, {input_name: obs_np})[0]
                # 把结果包回 tensor，为了完美衔接原代码第 29 行的返回逻辑
                action_tensor = torch.tensor(action_np, device=self.device)
            else:

                # --- 插入调试代码开始 ---
                print("\n" + "="*40)
                print(f"[DEBUG] 总观测 Tensor 形状: {obs_tensor.shape}")
                if isinstance(obs_buf_dict, dict):
                    for key, val in obs_buf_dict.items():
                        if hasattr(val, 'shape'):
                            print(f"  - {key}: {val.shape}")
                print("="*40 + "\n")
                # --- 插入调试代码结束 ---

                action_tensor = self.policy(obs_tensor) # 这是第 44 行报错的地方
                # ...

                # === 原版 PyTorch 推理模式 ===
                action_tensor = self.policy(obs_tensor)

        return action_tensor.detach().cpu().numpy().squeeze()

    def run(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            action = self._inference(obs_buf_dict)

            obs_buf_dict = self.env.step(action)
            
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()
            
    def run_eval(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            action = self._inference(obs_buf_dict)

            obs_buf_dict = self.env.step(action)
            
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()    

            if self.env.motion_loader.cur_motion_end:
                obs_buf_dict = self.env.next_motion()

        
