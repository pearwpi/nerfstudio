# Import necessary libraries
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pathlib import Path

import tyro
import multiprocessing

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
# from nerfstudio.viewer.new_custom_viewer_tello import Custom_Viewer
from nerfstudio.viewer.perspective import Custom_Viewer
# from nerfstudio.viewer.new_custom_viewer_vicon_odom_custom_pos_controller import Custom_Viewer
# from nerfstudio.viewer.new_custom_viewer_multithreading_with_positions import Custom_Viewer
# from nerfstudio.viewer.new_custom_viewer_multithreading_with_orientations_velocity_control import Custom_Viewer

# from dronekit import connect

@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: tyro.conf.Suppress[int] = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})

@dataclass
class RunCustomViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""

    def main(self) -> None:
        """Main function."""
        config, pipeline, _, step = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )
        _start_custom_viewer(config, pipeline, step)

def _start_custom_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    # Initialize the Vehicle object outside the process

    # Pass the vehicle object to Custom_Viewer
    custom_viewer = Custom_Viewer(config, pipeline)

    try:
        custom_viewer.execute()
    except KeyboardInterrupt:
        pass

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Ensure 'spawn' method is used for multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunCustomViewer]).main()

if __name__ == "__main__":
    entrypoint()
