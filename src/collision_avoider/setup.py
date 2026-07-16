import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'collision_avoider'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ventuno',
    maintainer_email='vkommera@ucsd.edu',
    description='Collision-avoidance controller and data-collection tools for the '
                'Create 3 base, driven by collision_classifier free/blocked scores.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collision_avoider = collision_avoider.controller:main',
            'data_collection = collision_avoider.data_collection:main',
            'movement_test = collision_avoider.movement_test:main',
        ],
    },
)
