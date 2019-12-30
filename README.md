# ROS robotic arm

This package contains files for my robotic arm similar to owi535 which I use in my ROS workspace.
This repository is intended to be a good starting point for working with owi535 in a ROS environment.
It contains a URDF file which is a model of the robot used in a gazebo simulation.
The parts of the model are Collada meshes cut and exported from existing STL models found [here](https://www.thingiverse.com/thing:2314339)
and [here](https://www.thingiverse.com/thing:1455455)

## Start working

To start working with the model clone the repository into the src folder of an existing catkin workspace, and type:

Launch the simulation in Gazebo by:
```roslaunch owi535 gazebo.launch```

Launch rqt control by:
```roslaunch owi535 rqt.launch```

Launch rviz by typing: 
```roslaunch owi535 rviz.launch```


## No warranty
The model is very inaccurate and the meshes are not intended to be used for manufacturing a real robot.


