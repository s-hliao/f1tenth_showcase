from setuptools import find_packages, setup
import os
from glob import glob

package_name = "depth_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="team10",
    maintainer_email="team10@cmu.edu",
    description="Real-time metric depth via DAv2 + RealSense anchor-row scale fusion",
    license="MIT",
    entry_points={
        "console_scripts": [
            "depth_fusion_node = depth_fusion.depth_fusion_node:main",
        ],
    },
)
