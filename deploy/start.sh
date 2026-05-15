# Origin GMT Evaluation
# HYDRA_FULL_ERROR=1 export LIBGL_ALWAYS_SOFTWARE=1 && python run.py --config-name=gmt sim=mujoco robot=g1_29dof \
#     mimic.policy.checkpoint=./data/model/gmt.onnx \
#     robot.control.viewer=True \
#     robot.control.real_time=True \
#     mimic.motion.loop=True \
#     mimic.motion.motion_path=./data/motion/dance1_subject1.npz

# GMT Evaluation: config-name has to be mosaic (been tested)
# HYDRA_FULL_ERROR=1 export LIBGL_ALWAYS_SOFTWARE=1 && python run.py --config-name=mosaic sim=mujoco robot=g1_29dof \
#     ++robot.control.viewer=True \
#     ++robot.control.real_time=True \


# GMT Record mp4
# HYDRA_FULL_ERROR=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python run.py --config-name=mosaic sim=mujoco robot=g1_29dof \
#   ++robot.control.viewer=False \
#   ++robot.control.real_time=False \
#   mimic.motion.loop=False \
#   env.config.record_video.enabled=True \
#   env.config.record_video.exit_on_complete=True \
#   env.config.record_video.output_dir=./videos \
#   env.config.record_video.fps=30 \
#   env.config.record_video.width=1280 \
#   env.config.record_video.height=720

HYDRA_FULL_ERROR=1 python scripts/robustness_validation/run_validation_mujoco.py \
  --robotbridge_root /home/chengyuxuan/RobotBridge \
  --motion "/home/chengyuxuan/RobotBridge/deploy/data/motion/Walking/amass_g1_jump-06-rope normal run fast-kudo_poses.npz" \
  --checkpoint /home/chengyuxuan/RobotBridge/deploy/data/model/model_27000.onnx \
  --output_dir verify/robustness_validation_mujoco/smoke \
  --epsilon_values 0.0 0.02 \
  --push_velocities 0.0 1.0 \
  --num_trials 2 \
  --perturbation_modes composite \
  --record_video