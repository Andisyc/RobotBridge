# python ./run.py --config-name=mosaic sim=mujoco

# python ./run.py --config-name=level_locomotion sim=mujoco

# HYDRA_FULL_ERROR=1 python run.py --config-name=gmt \
#     mimic.policy.checkpoint=/path/to/gmt.pt \
#     robot.control.viewer=False \
#     robot.control.real_time=False \
#     mimic.motion.motion_path=/path/to/motion/path

# HYDRA_FULL_ERROR=1 python run.py --config-name=eval \
#     mimic.policy.checkpoint=./data/model/gmt.onnx \
#     mimic.policy.use_estimator=False \
#     robot.control.viewer=True \
#     robot.control.real_time=True \
#     mimic.motion.motion_path=./data/motion/dance1_subject1.npz \
#     mimic.policy.history_length=5 \
#     mimic.policy.eval_mode=True \
#     mimic.motion.command_horizon=1

# GMT Evaluation
HYDRA_FULL_ERROR=1 export LIBGL_ALWAYS_SOFTWARE=1 && python run.py --config-name=gmt sim=mujoco robot=g1_29dof \
    mimic.policy.checkpoint=./data/model/gmt.onnx \
    robot.control.viewer=True \
    robot.control.real_time=True \
    mimic.motion.loop=True \
    mimic.motion.motion_path=./data/motion/dance1_subject1.npz
