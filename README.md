# Flight-ready Drift-Aware LiDAR-Intertial Odometry and Mapping with Self-correcting Maps

<p align="center">
  <img src="assets/img/IMG_2033.jpg" alt="Flight platform with labelled hardware components" />
</p>

[![ci][ci-badge]][ci-link]
[![Pixi][pixi-img]][pixi-link]
[![Watch the demo][youtube-badge]][youtube-video]

> [!NOTE]
> Built on [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) by HKU MARS Lab — please cite their work if you use this.

## Quick start

Prerequisites: [pixi](https://pixi.sh) installed.

```bash
git clone --recursive https://github.com/alvgaona/fr-lio
cd fr-lio
pixi run vcs-import   # fetch Livox-SDK2 + livox_ros_driver2 into ./deps
pixi run build        # colcon build (humble env by default)
```

Run with the default indoor config and RViz:

```bash
pixi shell
source install/setup.bash
ros2 launch fr_lio lio.launch.py
```

### Common launch overrides

```bash
# Switch to the outdoor profile
ros2 launch fr_lio lio.launch.py config_file:=$(pwd)/config/outdoors.yaml

# Compare against mocap ground truth (publishes /ground_truth/odom + /path)
ros2 launch fr_lio lio.launch.py mocap:=true rigid_body_name:=91

# Headless run inside a namespace
ros2 launch fr_lio lio.launch.py namespace:=drone1 rviz:=false
```

Use the `jazzy` pixi environment instead of the default humble:

```bash
pixi run -e jazzy build
```

## Citation

```bibtex
@software{gaona2026frlio,
  author  = {Gaona, Alvaro J. and Perez-Saura, David and Campoy, Pascual},
  title   = {Flight-ready Drift-Aware LiDAR-Inertial Odometry and Mapping with Self-correcting Maps},
  year    = {2026},
  url     = {https://github.com/alvgaona/fr-lio},
  version = {0.1.0}
}
```

[ci-badge]: https://github.com/alvgaona/fr-lio/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/alvgaona/fr-lio/actions/workflows/ci.yml
[pixi-img]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
[pixi-link]: https://pixi.sh
[youtube-badge]: https://img.shields.io/badge/YouTube-Watch%20demo-red?logo=youtube&logoColor=white
[youtube-video]: https://www.youtube.com/watch?v=mVYm7tcp8Lg
