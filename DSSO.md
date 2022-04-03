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

![](https://latex.codecogs.com/svg.latex?\Large&space;%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D%7Bj%7D%7D%7B%5Cpartial%20%5Cmathbf%7BC_K%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20u_j%7D%7B%5Cpartial%20f%7Bx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_j%7D%7B%5Cpartial%20f_%7By%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_j%7D%7B%5Cpartial%20c_%7Bx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_j%7D%7B%5Cpartial%20c_y%7D%20%5C%20%5Cfrac%7B%5Cpartial%20v_j%7D%7B%5Cpartial%20f_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_j%7D%7B%5Cpartial%20f_y%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_j%7D%7B%5Cpartial%20c_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_j%7D%7B%5Cpartial%20c_y%7D%5Cend%7Bbmatrix%7D)

In DSSO *Jpdc* is expanded from a 2x4 to a 2x14 matrix.  Between images in time(left to left) DSSO uses the same computation as DSO as both images are from the same camera. **p**<sub>j</sub> and **K** are always for the left camera. The additional elements of Jpdc are set to 0 for left-left image pairs as there is no information about the right camera's intrisics or relative pose.


For left-right pairs, **p**<sub>*j*</sub> becomes **p**<sub>*r*</sub> and the element of *Jpdc* are computed with respect to **K**<sub>*l*</sub> and **K**<sub>*r*</sub>:

![](https://latex.codecogs.com/svg.latex?\LARGE&space;%5Cbegin%7Bmatrix%7D%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_%7Br%7D%7D%7B%5Cpartial%20%5Cmathbf%7BC_%7BK_l%7D%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Blx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Bly%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Blx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Bly%7D%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Blx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Bly%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Blx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Bly%7D%7D%5Cend%7Bbmatrix%7D%20%26%20%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_%7Br%7D%7D%7B%5Cpartial%20%5Cmathbf%7BC_%7BK_r%7D%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Brx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Bry%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Brx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Bry%7D%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Brx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Bry%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Brx%7D%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Bry%7D%7D%5Cend%7Bbmatrix%7D%5Cend%7Bmatrix%7D)

Additionally **p**<sub>*r*</sub> WRT the left-right transform **Rt**<sub>rl</sub> is added as below. The derivertives are the same as for the left-left transform:

![](https://latex.codecogs.com/svg.latex?\Large&space;%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_%7Br%7D%7D%7B%5Cpartial%20%5Cmathbf%7BC_%7BRt_%7Blr%7D%7D%7D%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20R_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20R_y%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20R_z%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20t_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20t_y%7D%20%26%20%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20t_z%7D%20%5C%5C%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20R_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20R_y%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20R_z%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20t_x%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20r_y%7D%20%26%20%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20t_z%7D%5Cend%7Bbmatrix%7D)


Given a pixel position in the left image plane **p**<sub>l</sub>, the projected position in the right image plane **p**<sub>*r*</sub> is:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cmathbf%7Bp%7D_r%20%3D%20%5Cmathbf%7BK%7D_r%20%5Cmathbf%7Bp%7D%5En_r%20%3D%20%5Cmathbf%7BK%7D_r%20%5Cleft%20(%20%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cmathbf%7BR%7D_%7Brl%7D%5Cmathbf%7BK%7D%5E%7B-1%7D_l%5Cmathbf%7Bp%7D_l%20%2B%20%5Crho_r%5Cmathbf%7Bt%7D_%7Brl%7D%20%5Cright%20)%0A%7D"/>


The derivertive of **p**<sub>*r*</sub> WRT **C**<sub>**K**<sub>*l*</sub></sub> can be computed as:


<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_r%7D%7B%5Cpartial%20%5Cmathbf%7BC_K%7D_l%7D%20%3D%0A%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_r%7D%7B%5Cpartial%20%5Cmathbf%7Bp%7D%5En_r%7D%20%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D%5En_r%7D%7B%5Cpartial%20%5Cmathbf%7BC_K%7D_l%7D%20%0A%7D"/>

Where **p**<sub>*r*</sub> WRT **p**<sub>*r*</sub><sup>n</sup> is:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cbegin%7Baligned%7D%0A%5Cmathbf%7Bp%7D_r%20%3D%20%5Cmathbf%7BK%7D%20%5Cmathbf%7Bp%7D%5En_r%20%26%3D%0A%5Cbegin%7Bbmatrix%7Df_x%20u%5En_r%20%2B%20c_x%20%5C%5C%20f_y%20v%5En_r%20%2B%20c_y%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_r%7D%7B%5Cpartial%20%5Cmathbf%7Bp%7D%5En_r%7D%20%26%3D%0A%5Cbegin%7Bbmatrix%7Df_x%20%26%200%20%26%200%20%5C%5C%200%20%26%20f_y%20%26%200%20%5C%5C%200%20%26%200%20%26%200%5Cend%7Bbmatrix%7D%0A%5Cend%7Baligned%7D%0A%7D"/>

.. and **p**<sub>*r*</sub><sup>n</sup> WRT **C**<sub>**K**<sub>*l*</sub></sub> is(see Tong's blog):

<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20u_r%5En%7D%7B%5Cpartial%20f_%7Blx%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20u_r%5En-r_%7B00%7D%5Cright)%20f_%7Blx%7D%5E%7B-2%7D%5Cleft(u_l-c_%7Blx%7D%5Cright)%20%5C%5C%20%0A%5Cfrac%7B%5Cpartial%20u_r%5En%7D%7B%5Cpartial%20f_%7Bly%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20u_r%5En-r_%7B01%7D%5Cright)%20f_%7Bly%7D%5E%7B-2%7D%5Cleft(v_l-c_%7Bly%7D%5Cright)%5C%5C%0A%5Cfrac%7B%5Cpartial%20u_r%5En%7D%7B%5Cpartial%20c_%7Blx%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20u_r%5En-r_%7B00%7D%5Cright)%20f_%7Blx%7D%5E%7B-1%7D%20%5C%5C%20%0A%5Cfrac%7B%5Cpartial%20u_r%5En%7D%7B%5Cpartial%20c_%7Bly%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20u_r%5En-r_%7B01%7D%5Cright)%20f_%7Bly%7D%5E%7B-1%7D%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%5En%7D%7B%5Cpartial%20f_%7Blx%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20v_r%5En-r_%7B10%7D%5Cright)%20f_%7Blx%7D%5E%7B-2%7D%5Cleft(u_l-c_%7Blx%7D%5Cright)%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%5En%7D%7B%5Cpartial%20f_%7Bly%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20v_r%5En-r_%7B11%7D%5Cright)%20f_%7Bly%7D%5E%7B-2%7D%5Cleft(v_l-c_%7Bly%7D%5Cright)%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%5En%7D%7B%5Cpartial%20c_%7Blx%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20v_r%5En-r_%7B10%7D%5Cright)%20f_%7Blx%7D%5E%7B-1%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%5En%7D%7B%5Cpartial%20c_%7Bly%7D%7D%26%3D%0A%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20v_r%5En-r_%7B11%7D%5Cright)%20f_%7Bly%7D%5E%7B-1%7D%0A%5Cend%7Baligned%7D%0A%7D"/>


So for ![](https://latex.codecogs.com/svg.latex?\Large&space;%7B%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_r%7D%7B%5Cpartial%20%5Cmathbf%7BC_K%7D_l%7D%7D) we get:


<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cbegin%7Baligned%7D%0A%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Blx%7D%7D%26%3Df_%7Brx%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20u_r%5En-r_%7B00%7D%5Cright)%20f_%7Blx%7D%5E%7B-2%7D%5Cleft(u_l-c_%7Blx%7D%5Cright)%5C%5C%20%0A%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20f_%7Bly%7D%7D%26%3Df_%7Brx%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20u_r%5En-r_%7B01%7D%5Cright)%20f_%7Bly%7D%5E%7B-2%7D%5Cleft(v_l-c_%7Bly%7D%5Cright)%5C%5C%0A%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Blx%7D%7D%26%3Df_%7Brx%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20u_r%5En-r_%7B00%7D%5Cright)%20f_%7Blx%7D%5E%7B-1%7D%5C%5C%20%0A%5Cfrac%7B%5Cpartial%20u_r%7D%7B%5Cpartial%20c_%7Bly%7D%7D%26%3Df_%7Brx%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20u_r%5En-r_%7B01%7D%5Cright)%20f_%7Bly%7D%5E%7B-1%7D%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Blx%7D%7D%26%3Df_%7Bry%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20v_r%5En-r_%7B10%7D%5Cright)%20f_%7Blx%7D%5E%7B-2%7D%5Cleft(u_l-c_%7Blx%7D%5Cright)%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20f_%7Bly%7D%7D%26%3Df_%7Bry%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20v_r%5En-r_%7B11%7D%5Cright)%20f_%7Bly%7D%5E%7B-2%7D%5Cleft(v_l-c_%7Bly%7D%5Cright)%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Blx%7D%7D%26%3Df_%7Bry%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B20%7D%20v_r%5En-r_%7B10%7D%5Cright)%20f_%7Blx%7D%5E%7B-1%7D%5C%5C%0A%5Cfrac%7B%5Cpartial%20v_r%7D%7B%5Cpartial%20c_%7Bly%7D%7D%26%3Df_%7Bry%7D%5Cfrac%7B%5Crho_r%7D%7B%5Crho_l%7D%5Cleft(r_%7B21%7D%20v_r%5En-r_%7B11%7D%5Cright)%20f_%7Bly%7D%5E%7B-1%7D%0A%5Cend%7Baligned%7D%0A%7D"/>


As **p**<sup>n</sup><sub>*r*</sub> doesn't depend on **C**<sub>**K**<sub>*r*</sub></sub> the derivertive of **p**<sub>*r*</sub> WRT **C**<sub>**K**<sub>*r*</sub></sub> is **p**<sub>*r*</sub> WRT **p**<sub>*r*</sub><sup>n</sup> with *u*<sup>n</sup><sub>r</sub> and *v*<sup>n</sup><sub>r</sub> substituted:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;%7B%0A%5Cbegin%7Baligned%7D%0A%5Cmathbf%7Bp%7D_r%20%3D%20%5Cmathbf%7BK%7D%20%5Cmathbf%7Bp%7D%5En_r%20%26%3D%0A%5Cbegin%7Bbmatrix%7Df_x%20u%5En_r%20%2B%20c_x%20%5C%5C%20f_y%20u%5En_r%20%2B%20c_y%5C%5C%201%20%5Cend%7Bbmatrix%7D%20%5C%5C%0A%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bp%7D_r%7D%7B%5Cpartial%20%5Cmathbf%7BK%7D%7D%20%26%3D%0A%5Cbegin%7Bbmatrix%7Du%5En_r%20%26%200%20%26%201%20%26%200%20%5C%5C%200%20%26%20v%5En_r%20%26%200%20%26%201%20%5C%5C%200%20%26%200%20%26%200%20%26%200%5Cend%7Bbmatrix%7D%0A%5Cend%7Bbmatrix%7D%0A%7D"/>







