# GMT Evaluation: config-name has to be mosaic (been tested)
HYDRA_FULL_ERROR=1 export LIBGL_ALWAYS_SOFTWARE=1 && python run.py --config-name=mosaic sim=mujoco robot=g1_29dof \
    mimic.policy.checkpoint=./data/model/gmt.onnx \
    robot.control.viewer=True \
    robot.control.real_time=True \
    mimic.motion.loop=True \
    mimic.motion.motion_path=./data/motion/dance1_subject1.npz
