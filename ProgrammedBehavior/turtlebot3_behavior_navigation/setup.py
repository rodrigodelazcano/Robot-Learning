from setuptools import setup

package_name = 'turtlebot3_behavior_navigation'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/world_launch.py']))
data_files.append(('share/' + package_name + '/worlds', [
    'worlds/indoors_world.wbt',
]))
data_files.append(('share/' + package_name, ['package.xml']))
# data_files.append(('share/ament_index/resource_index/packages',
#             ['resource/' + package_name]),
#         ('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rodri',
    maintainer_email='r.l.p.v96@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'random_navigation = turtlebot3_behavior_navigation.random_navigation:main',
                'teleop = turtlebot3_behavior_navigation.teleop:main ',
        ],
    },
)
