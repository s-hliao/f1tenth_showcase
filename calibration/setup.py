from setuptools import find_packages, setup
import os
from glob import glob

package_name = "calibration"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team10",
    maintainer_email="yuansu@andrew.cmu.edu",
    description="Targetless LiDAR-camera extrinsic calibration using natural scenes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "collect_node = calibration.collect_node:main",
        ],
    },
)
