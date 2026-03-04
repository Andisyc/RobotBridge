# RobotBridge

🤖 Unified Sim2Sim and Sim2Real Deployment Framework for Humanoid Robots - Plug and Play

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://opensource.org/license/apache-2-0)

## ✨ Features

- 🎯 **Unified Interface**: Same code runs in simulation and on real robots
- 🔄 **Policy Switching**: Real-time smooth switching between locomotion and mimic policies
- 🎮 **Multiple Control Methods**: Keyboard, joystick, and programmatic control
- 🤖 **Multi-Robot Support**: Unitree G1, H1, H1-2, Adam Lite, and more
- 📊 **Visualization Tools**: Built-in MuJoCo visualization and motion markers
- 🔧 **Flexible Configuration**: Hydra-based configuration management system
- 📈 **Smooth Interpolation**: Automatic pose interpolation during policy transitions

## 📦 Installation

### Prerequisites

- Python 3.8+
- GCC/G++ compiler (required for real robot deployment)
- CUDA (optional, for GPU acceleration)

### Step 1: Create Conda Environment

```bash
conda create -n rb python=3.8 -y
conda activate rb
```

### Step 2: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/BAAI-Humanoid/RobotBridge.git
cd RobotBridge

# Install Python dependencies
pip install -r requirements.txt

```

### Step 3: Compile Transition Layer (Real Robot Only)

```bash
cd unitree_sdk2
find . -exec touch -c {} \;  # Update file timestamps
mkdir -p build && cd build
cmake ..
make
```

After compilation, you'll find the `trans` executable in the `bin/` directory.

## 🚀 Quick Start

### Simulation Testing

#### 1. Locomotion

```bash
# level is just our locomotion policy, you can train yours using any open-sourced codes freely
cd deploy
python run.py --config-name=level_locomotion 
```

**Controls:**
- `W/S`: Move forward/backward
- `A/D`: Strafe left/right
- `Q/E`: Rotate left/right
- `Space`: Stop

#### 2. Motion Mimic

```bash
# mosaic style
cd deploy
python run.py --config-name=mosaic \
    env.config.motion.motion_path=data/motion/your_motion.npz \
    env.config.policy.checkpoint=data/model/your_model.onnx
```

#### 3. Locomotion + Motion Mimic Policy Switching

```bash
cd deploy
python run.py --config-name=loco_mimic
```

**Controls:**
- `W/A/S/D/Q/E`: Movement control (in locomotion mode)
- `L`: Switch to Locomotion policy
- `K`: Switch to Mimic policy
- `Space`: Stop movement

### Real Robot Deployment (Unitree G1)

**Important Notes:**
- ⚠️ Must follow Unitree's recommendations for safety while using robots in low-level control (debug) mode
- ⚠️ Must launch transition layer before policy layer
- ⚠️ Ensure network connection is working (192.168.123.x subnet)
- ⚠️ Test in a safe environment for first deployment
- ⚠️ Have an emergency stop button ready
- ⚠️ This is research code; use at your own risk; we do not take responsibility for any damage.

#### Step 1: Launch Transition Layer

On the robot, execute:

```bash
cd unitree_sdk2/build/bin

# Check network interface
ifconfig  # Find the interface name corresponding to 192.168.123.164 (e.g., eth0 or eth1)

# Launch transition layer
./trans_wo_lock eth0

# Press ENTER to trigger communication
# When the transition layer is runing, press [L2 + B] to stop the process,
#                                      press [L2 + Y] to recover from stopped state
#                                      press [L1 + B] for emergency termination.
```

#### Step 2: Launch Policy Layer

In another terminal or remote host:

```bash
cd deploy
python run.py --config-name=mosaic \
    simulator=real_world \
    device=cpu

# Press R2 on joystick to enter prepare mode, press again to enter the policy runing mode
# When policy is runing, pressing R2 the agent will return to prepare mode.
```

#### Step 3: Joystick Control

On the real robot, use the Unitree joystick:

**Stick Controls (Locomotion):**
- Left stick: Forward/backward and left/right movement (vx, vy)
- Right stick: Rotation (yaw)

**Policy Switching Buttons (LocoMimic):**
- `L1` (left upper button): Switch to Locomotion policy
- `L2` (left lower left button): Switch to Mimic policy

### Teleoperation
Refer to [MOSAIC-teleop](https://github.com/BAAI-Humanoid/MOSAIC-teleop.git) and deploy the specified environment on the local PC.


## 📚 Configuration Guide

### Configuration File Organization

RobotBridge uses Hydra for configuration management with a modular design:

```yaml
# config/loco_mimic.yaml
defaults:
  - _self_
  - robot: g1_29dof  # Full 29-DoF robot configuration
  - obs: level       # Observation configuration (will be overridden per policy)
  - sim: mujoco      # mujoco or real_world
  - agent: loco_mimic
  - env: loco_mimic_switch
  - mimic: mosaic
  - locomotion: level
  - teleop: teleop

device: 'cuda'
```

### Command Line Parameter Override

Use Hydra syntax to override configurations:

```bash
# Change device
python run.py --config-name=loco_mimic device=cuda

# Change robot
python run.py --config-name=loco_mimic \
    asset=h1_19dof \
    control=h1_19dof \
    robot=h1_19dof

# Change motion file
python run.py --config-name=mosaic \
    env.config.motion.motion_path=data/motion/custom_motion.npz

# Disable visualization markers
python run.py --config-name=level_locomotion \
    simulator.config.marker=false
```

## 🏗️ System Architecture

```
RobotBridge/
├── deploy/                      # Policy Layer
│   ├── agents/                  # Agent implementations
│   │   ├── base_agent.py
│   │   ├── level_agent.py
│   │   ├── mosaic_agent.py
│   │   └── loco_mimic_agent.py  # Policy switching agent
│   ├── envs/                    # Environment implementations
│   │   ├── base_env.py
│   │   ├── level_locomotion.py
│   │   ├── mosaic.py
│   │   └── loco_mimic_switch.py # Policy switching environment
│   ├── simulator/               # Simulator interfaces
│   │   ├── mujoco.py            # MuJoCo simulation
│   │   └── real_world.py        # Real robot interface
│   ├── config/                  # Configuration files
│   └── utils/                   # Utility functions
├── unitree_sdk2/                # Transition Layer
│   ├── trans.cpp                # C++ communication layer
│   └── lcm_types/               # LCM message definitions
```

## 🤖 Supported Robots

| Robot | DoF | Config File |
|-------|-----|------------|
| Unitree G1 (12 DoF) | 12 | `g1_29dof_anneal_12dof.yaml` |
| Unitree G1 (23 DoF) | 23 | `g1_29dof_anneal_23dof.yaml` |
| Unitree G1 (29 DoF) | 29 | `g1_29dof.yaml` |
| Unitree H1 | 19 | `h1_19dof.yaml` |
| Unitree H1-2 | 21 | `h1_2_27dof_anneal_21dof.yaml` |
| Adam Lite AGX | 23 | `adam_lite_agx_23dof.yaml` |

## 🐛 Troubleshooting

### Common Issues

#### 1. MuJoCo Display Issues

**Issue**: `GLEW initialization error`

**Solution**:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

#### 2. Real Robot Connection Failed

**Issue**: LCM messages not received

**Solutions**:
- Check network connection: `ping 192.168.123.164`
- Confirm transition layer is running and ENTER was pressed
- Use monitor tool: `python deploy/monitor_lcm.py`

#### 3. Joystick Buttons Not Responding

**Issue**: Pressing L1/L2 buttons doesn't switch policies

**Possible Causes and Solutions**:
1. **LCM messages not received**: Run `python deploy/monitor_lcm.py` and press buttons to check for messages
2. **Incorrect button mapping**: Verify correct buttons are used (L1=left_upper_switch, L2=left_lower_left_switch)
3. **Configuration issue**: Confirm using `--config-name=loco_mimic`

#### 4. Abnormal Pose During Policy Switch

**Issue**: Strange pose when switching from locomotion to mimic

**Fixed**: Ensure you're using the latest version. DoF order conversion and default pose issues have been resolved.

#### 5. Import Errors

**Issue**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure you're in the correct directory
cd deploy
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 6. LCM Issues

**Issue**: robot can't receive tele-operation LCM messages from local PC

**Solutions**:
1. Local PC and robot need to be on the same local area network, you can check by `ping` each other
2. Specify the network port of `239.255.0.0` by running `sudo ip route add 239.255.0.0/16 dev {network port}`
  - You can check network port by running `ifconfig` on Linux
3. If there is already sepcified network port, delete it first by running `sudo ip route del 239.255.0.0/16 dev {old network port}`
4. You can check the result by running `ip route | grep 239`, and if you can see `239.255.0.0/16 dev {network port} scope link`, you succeed.

## 📌 Evaluation Guide
#### Overview
This guide outlines the standard procedure for running model evaluation with automatic checkpoint loading based on file extensions (ONNX/TorchScript).

#### Prerequisites
- Ensure all dependencies are installed as per the project requirements.
- Valid checkpoint files (`.onnx`, `.pt`, `.pth`, `.jit`) and motion data are prepared at `/path/to/data`.

#### Standard Evaluation
1. Modify the `eval.sh` script with the following core configurations (adjust parameters as needed):
   ```bash
   HYDRA_FULL_ERROR=1 python run.py --config-name=eval \
       mimic.policy.checkpoint=/path/to/checkpoint.onnx \
       mimic.policy.use_estimator=False \
       robot.control.viewer=False \
       robot.control.real_time=False \
       mimic.motion.motion_path=/path/to/motion/path \
       mimic.policy.history_length=5 \
       mimic.policy.eval_mode=True \
       mimic.motion.command_horizon=1
   ```
2. Optimize evaluation speed by modifying `deploy/envs/mosaic.py`:
   - Locate the function with `save_video` enabled.
   - Keep only `self.frames = []` and comment out all other related lines.

#### Specialized Evaluation (GMT/Twist)
Use the following commands directly for GMT and Twist model evaluation (adjust motion path if necessary):
```bash
# GMT Evaluation
HYDRA_FULL_ERROR=1 python run.py --config-name=gmt \
    mimic.policy.checkpoint=/path/to/gmt.pt \
    robot.control.viewer=False \
    robot.control.real_time=False \
    mimic.motion.motion_path=/path/to/motion/path

# Twist Evaluation
HYDRA_FULL_ERROR=1 python run.py --config-name=twist \
    mimic.policy.checkpoint=/path/to/twist.pt \
    robot.control.viewer=False \
    robot.control.real_time=False \
    mimic.motion.motion_path=/path/to/motion/path
```

#### Notes
- All file paths (checkpoint/motion data) should be replaced with actual paths in production use.
- `viewer` and `real_time` are set to `False` by default to maximize evaluation efficiency.
- Adjust `history_length` and `command_horizon` according to the actual model configuration.

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- This project is build upon [Walk these ways](https://github.com/Improbable-AI/walk-these-ways)
- Unitree Robotics for robot hardware and SDK
- MuJoCo for physics simulation
- Hydra for configuration management

## 📞 Contact

For questions or suggestions, please submit an Issue or contact the maintainers.

---