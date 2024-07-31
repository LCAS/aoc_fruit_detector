from setuptools import find_packages, setup

package_name = 'aoc_fruit_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='A.Yilmaz',
    maintainer_email='ayilmaz@lincoln.ac.uk',
    description='AOC Fruit Detector package, part of Agri-OpenCore (AOC) project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
