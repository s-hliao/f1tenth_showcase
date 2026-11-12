import os
from glob import glob

from setuptools import setup

package_name = 'showcase_fleet'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name, package_name + '.offline'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', [p for p in glob('config/*') if os.path.isfile(p)]),
        ('share/' + package_name + '/config/tracks', glob('config/tracks/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/scripts', [p for p in glob('scripts/*') if os.path.isfile(p)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DRIVE Lab',
    maintainer_email='hzl@andrew.cmu.edu',
    description='Showcase fleet: car-side broadcaster/listener (relay, drive gate, carctl) and laptop '
                'listener/broadcaster (LLA estimator, ground-station GUI), plus DDS isolation configs',
    license='MIT',
    entry_points={
        'console_scripts': [
            # car
            'drive_gate = showcase_fleet.drive_gate:main',
            'fleet_relay = showcase_fleet.fleet_relay:main',
            'carctl = showcase_fleet.carctl:main',
            'lqr = showcase_fleet.lqr_cli:main',
            # laptop
            'ground_station = showcase_fleet.ground_station:main',
            'lla_estimator = showcase_fleet.lla_estimator:main',
            'viz_ref_fiala = showcase_fleet.offline.viz_ref_fiala:main',
            # desk test
            'fake_car = showcase_fleet.fake_car:main',
        ],
    },
)
