1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

Parameterised copy: `robotnik_arm_collect.py`
-------------------------------------------
The original script is unchanged. From `final/`, run one scene with:

```bash
ROBOENV_HEADLESS=1 python robotnik_arm_task/robotnik_arm_collect.py \
  --scenarios artifacts/scenarios.json --scene-id scene_001 --output artifacts/trajectories
```

The JSON has a `scenarios` list. Each scene has `id`, `target`, and optionally
`base_pos`, `base_bearing`, `arm_q`, `arm_qd`, and `seed`. Omitted initial conditions
use the original robot config. The robot config itself is not modified.
Without `ROBOENV_HEADLESS=1`, the original GUI and plots remain enabled.
`ROBOENV_MAX_STEPS` retains the original step-cap behaviour and stop-check order.

Each scene produces `<id>.npz` (existing files are not overwritten), containing
`inputs` (N x 21: base xyz, yaw, arm q, arm qd, target xyz), `actions` (N x 11:
four wheel velocities and seven arm torques), `time`, `ee_pos`, `phase`,
`final_state`, `final_ee_pos`, and JSON strings `scenario` and `metadata`.
Metadata contains only `success`, `stop_reason`, and `final_error_m`.
Load with `numpy.load(path, allow_pickle=False)` and `json.loads(str(data['metadata']))`.
Every recorded input is paired with its pre-step expert action; terminal metrics
are refreshed after the final action. Expert controller functions are shared with
the supplied script, and the original plotting/stop logic is retained.

Batch collection (from `final/`):

```bash
python collect_data.py --scenarios artifacts/scenarios.json --output artifacts/collection
```

`collect_data.py` launches this parameterised script once per scene using the same
Python interpreter. It defaults to headless mode; `--gui` enables the original
window and plots (close the plots to proceed to the next scene). `--max-steps`
passes the step cap to the child script. Resources are read beside the child
script in `robotnik_arm_task/configs/` and `robotnik_arm_task/models/`;
the old `--resource-root` option is no longer used.

The batch output directory must be new. It contains a copy of `scenarios.json`,
each scene's `.npz`, and `summary.csv` with `id`, `success`, `stop_reason`, and
`final_error_m`. Successful and unsuccessful completed trajectories are both
saved. A child process error stops the batch; previous results remain saved.
