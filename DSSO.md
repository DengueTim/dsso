# DSSO: Direct Sparse Stereo Odometry

**WIP** 

An attempt at extending DSO to stereo image pairs. The intention is to remove scale abiguity by using a fixed stereo baseline and to improve initialisation and initial estimates using the image pairs.

Ideas/approach:
 - [x] Change FrameHessian to have left and right images with shared(but offset) pose...
 - [ ] Assuming a stereo camera uses the same exposure for both images determine if/how to share a and b illumination params.
 - [ ] Keep number of points per image or share same number of points between image pairs?
 - [ ] ...

For more information see
[https://vision.in.tum.de/dso](https://vision.in.tum.de/dso)

### Maths...

See [Tong Ling's DSO Blog posts](https://tongling916.github.io/tags/#DSO) for analysis of the DSO code and it's maths.


#### Jacobian - Intrinsic and Left-Right Pose Parameters Additions

Elements of *RawResidualJacobian.Jpdc* in DSO:

![](https://latex.codecogs.com/svg.latex?\LARGE&space;\frac{\partial \mathbf{p}_{j}}{\partial \mathbf{C_K}} = \begin{bmatrix} \frac{\partial u_j}{\partial f_{x}} & \frac{\partial u_j}{\partial f_{y}} & \frac{\partial u_j}{\partial c_{x}} & \frac{\partial u_j}{\partial c_y} \\ \frac{\partial v_j}{\partial f_x} & \frac{\partial v_j}{\partial f_y} & \frac{\partial v_j}{\partial c_x} & \frac{\partial v_j}{\partial c_y}\end{bmatrix})

In DSSO *Jpdc* is expanded from a 2x4 to a 2x14 matrix.  Between images in time(left to left) DSSO uses the same computation as DSO as both images are from the same camera. **p**<sub>j</sub> and **K** are always for the left camera. The additional elements of Jpdc are set to 0 for left-left image pairs as there is no information about the right camera's intrisics or relative pose.


For left-right pairs, **p**<sub>*j*</sub> becomes **p**<sub>*r*</sub> and the element of *Jpdc* are computed with respect to **K**<sub>*l*</sub> and **K**<sub>*r*</sub>:

![](https://latex.codecogs.com/svg.latex?\LARGE&space;\begin{matrix}\frac{\partial \mathbf{p}_{r}}{\partial \mathbf{C_{K_l}}} = \begin{bmatrix}\frac{\partial u_r}{\partial f_{lx}} & \frac{\partial u_r}{\partial f_{ly}} & \frac{\partial u_r}{\partial c_{lx}} & \frac{\partial u_r}{\partial c_{ly}} \\ \frac{\partial v_r}{\partial f_{lx}} & \frac{\partial v_r}{\partial f_{ly}} & \frac{\partial v_r}{\partial c_{lx}} & \frac{\partial v_r}{\partial c_{ly}}\end{bmatrix} & \frac{\partial \mathbf{p}_{r}}{\partial \mathbf{C_{K_r}}} = \begin{bmatrix}\frac{\partial u_r}{\partial f_{rx}} & \frac{\partial u_r}{\partial f_{ry}} & \frac{\partial u_r}{\partial c_{rx}} & \frac{\partial u_r}{\partial c_{ry}} \\ \frac{\partial v_r}{\partial f_{rx}} & \frac{\partial v_r}{\partial f_{ry}} & \frac{\partial v_r}{\partial c_{rx}} & \frac{\partial v_r}{\partial c_{ry}}\end{bmatrix}\end{matrix})

Additionally **p**<sub>*r*</sub> WRT the left-right transform **Rt**<sub>rl</sub> is added as below. The derivertives are the same as for the left-left transform:

![](https://latex.codecogs.com/svg.latex?\LARGE&space;\frac{\partial \mathbf{p}_{r}}{\partial \mathbf{C_{Rt_{lr}}}} = \begin{bmatrix}\frac{\partial u_r}{\partial R_x} & \frac{\partial u_r}{\partial R_y} & \frac{\partial u_r}{\partial R_z} & \frac{\partial u_r}{\partial t_x} & \frac{\partial u_r}{\partial t_y} & \frac{\partial u_r}{\partial t_z} \\ \frac{\partial v_r}{\partial R_x} & \frac{\partial v_r}{\partial R_y} & \frac{\partial v_r}{\partial R_z} & \frac{\partial v_r}{\partial t_x} & \frac{\partial v_r}{\partial r_y} & \frac{\partial v_r}{\partial t_z}\end{bmatrix})




Given a pixel position in the left image plane **p**<sub>l</sub>, the projected position in the right image plane **p**<sub>*r*</sub> is:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\mathbf{p}_r = \mathbf{K}_r \mathbf{p}^n_r = \mathbf{K}_r \left ( \frac{\rho_r}{\rho_l}\mathbf{R}_{rl}\mathbf{K}^{-1}_l\mathbf{p}_l + \rho_r\mathbf{t}_{rl} \right )
}"/>


The derivertive of **p**<sub>*r*</sub> WRT **C**<sub>**K**<sub>*l*</sub></sub> can be computed as:


<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\frac{\partial \mathbf{p}_r}{\partial \mathbf{C_K}_l} =
\frac{\partial \mathbf{p}_r}{\partial \mathbf{p}^n_r} \frac{\partial \mathbf{p}^n_r}{\partial \mathbf{C_K}_l} 
}"/>

Where **p**<sub>*r*</sub> WRT **p**<sub>*r*</sub><sup>n</sup> is:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\begin{aligned}
\mathbf{p}_r = \mathbf{K} \mathbf{p}^n_r &=
\begin{bmatrix}f_x u^n_r + c_x \\ f_y v^n_r + c_y\\ 1 \end{bmatrix} \\
\frac{\partial \mathbf{p}_r}{\partial \mathbf{p}^n_r} &=
\begin{bmatrix}f_x & 0 & 0 \\ 0 & f_y & 0 \\ 0 & 0 & 0\end{bmatrix}
\end{aligned}
}"/>

.. and **p**<sub>*r*</sub><sup>n</sup> WRT **C**<sub>**K**<sub>*l*</sub></sub> is(see Tong's blog):

<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\begin{aligned}
\frac{\partial u_r^n}{\partial f_{lx}}&=
\frac{\rho_r}{\rho_l}\left(r_{20} u_r^n-r_{00}\right) f_{lx}^{-2}\left(u_l-c_{lx}\right) \\ 
\frac{\partial u_r^n}{\partial f_{ly}}&=
\frac{\rho_r}{\rho_l}\left(r_{21} u_r^n-r_{01}\right) f_{ly}^{-2}\left(v_l-c_{ly}\right)\\
\frac{\partial u_r^n}{\partial c_{lx}}&=
\frac{\rho_r}{\rho_l}\left(r_{20} u_r^n-r_{00}\right) f_{lx}^{-1} \\ 
\frac{\partial u_r^n}{\partial c_{ly}}&=
\frac{\rho_r}{\rho_l}\left(r_{21} u_r^n-r_{01}\right) f_{ly}^{-1}\\
\frac{\partial v_r^n}{\partial f_{lx}}&=
\frac{\rho_r}{\rho_l}\left(r_{20} v_r^n-r_{10}\right) f_{lx}^{-2}\left(u_l-c_{lx}\right) \\
\frac{\partial v_r^n}{\partial f_{ly}}&=
\frac{\rho_r}{\rho_l}\left(r_{21} v_r^n-r_{11}\right) f_{ly}^{-2}\left(v_l-c_{ly}\right)\\
\frac{\partial v_r^n}{\partial c_{lx}}&=
\frac{\rho_r}{\rho_l}\left(r_{20} v_r^n-r_{10}\right) f_{lx}^{-1} \\
\frac{\partial v_r^n}{\partial c_{ly}}&=
\frac{\rho_r}{\rho_l}\left(r_{21} v_r^n-r_{11}\right) f_{ly}^{-1}
\end{aligned}
}"/>


So for ![](https://latex.codecogs.com/svg.latex?\Large&space;{\frac{\partial \mathbf{p}_r}{\partial \mathbf{C_K}_l}}) we get:


<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\begin{aligned}
\frac{\partial u_r}{\partial f_{lx}}&=f_{rx}
\frac{\rho_r}{\rho_l}\left(r_{20} u_r^n-r_{00}\right) f_{lx}^{-2}\left(u_l-c_{lx}\right) \\ 
\frac{\partial u_r}{\partial f_{ly}}&=f_{rx}
\frac{\rho_r}{\rho_l}\left(r_{21} u_r^n-r_{01}\right) f_{ly}^{-2}\left(v_l-c_{ly}\right)\\
\frac{\partial u_r}{\partial c_{lx}}&=f_{rx}
\frac{\rho_r}{\rho_l}\left(r_{20} u_r^n-r_{00}\right) f_{lx}^{-1} \\ 
\frac{\partial u_r}{\partial c_{ly}}&=f_{rx}
\frac{\rho_r}{\rho_l}\left(r_{21} u_r^n-r_{01}\right) f_{ly}^{-1}\\
\frac{\partial v_r}{\partial f_{lx}}&=f_{ry}
\frac{\rho_r}{\rho_l}\left(r_{20} v_r^n-r_{10}\right) f_{lx}^{-2}\left(u_l-c_{lx}\right) \\
\frac{\partial v_r}{\partial f_{ly}}&=f_{ry}
\frac{\rho_r}{\rho_l}\left(r_{21} v_r^n-r_{11}\right) f_{ly}^{-2}\left(v_l-c_{ly}\right)\\
\frac{\partial v_r}{\partial c_{lx}}&=f_{ry}
\frac{\rho_r}{\rho_l}\left(r_{20} v_r^n-r_{10}\right) f_{lx}^{-1} \\
\frac{\partial v_r}{\partial c_{ly}}&=f_{ry}
\frac{\rho_r}{\rho_l}\left(r_{21} v_r^n-r_{11}\right) f_{ly}^{-1}
\end{aligned}
}"/>


As **p**<sup>n</sup><sub>*r*</sub> doesn't depend on **C**<sub>**K**<sub>*r*</sub></sub> the derivertive of **p**<sub>*r*</sub> WRT **C**<sub>**K**<sub>*r*</sub></sub> is **p**<sub>*r*</sub> WRT **p**<sub>*r*</sub><sup>n</sup> with *u*<sup>n</sup><sub>r</sub> and *v*<sup>n</sup><sub>r</sub> substituted:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;{
\begin{aligned}
\mathbf{p}_r = \mathbf{K} \mathbf{p}^n_r &=
\begin{bmatrix}f_x u^n_r + c_x \\ f_y u^n_r + c_y\\ 1 \end{bmatrix} \\
\frac{\partial \mathbf{p}_r}{\partial \mathbf{K}} &=
\begin{bmatrix}u^n_r & 0 & 1 & 0 \\ 0 & v^n_r & 0 & 1 \\ 0 & 0 & 0 & 0\end{bmatrix}
\end{bmatrix}
}"/>







