from glob import glob

from setuptools import setup

package_name = 'showcase_state_estimation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DRIVE Lab',
    maintainer_email='hzl@andrew.cmu.edu',
    description='EKF state estimation (robot_localization) for the showcase cars, ported from llampc',
    license='MIT',
    entry_points={
        'console_scripts': [
            'imu_zupt_prep = showcase_state_estimation.imu_zupt_prep:main',
            'optitrack_node = showcase_state_estimation.optitrack_node:main',
            'odom_points = showcase_state_estimation.odom_points:main',
            'imu_check = showcase_state_estimation.imu_check:main',
        ],
    },
)
