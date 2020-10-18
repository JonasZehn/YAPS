# YAPS - Yet another particle splatter

This is a particle splatter intended for the rendering of smoke.
The input is a list of velocity grid files specifically .uni files, which can be generated using e. g. [mantaflow](http://mantaflow.com/).
Particles are then traced through these velocity fields and rendered.
The particles are illuminated by rendering the particles from the camera in sorted order to allow for a simple model based on transparency.
Then the particles are rendered to the image plane.
In practice a non-realistic/artistic color transition can be used based on the computed illumination value.

## Compilation

The project can be compiled using cmake, but the following libraries are required:

* [Eigen](http://eigen.tuxfamily.org)
* [zlib](https://zlib.net/): Used to load the .uni files
* [Qt 5](https://www.qt.io/)
* CUDA
* libpng

## Usage

To generate some example data, compile mantaflow and then run the example file inside the *manta-scenes* folder using
```
$ manta plume.py mc
```
This will write velocity grids to the folder *manta-scenes/out/plume_mc_128* which then can be loaded in YAPS and used to render the particles.
The particle instantiation is currently done in the C++ code and thus has to be changed manually.
When running the program multiple parameters related to lighting can be chosen in the GUI. Images are automatically written to disk, which then can be combined to a video file using e. g. [ffmpeg](https://ffmpeg.org/)

```
$ ffmpeg.exe -y -f image2 -framerate 30 -i image_%04d.png video.mp4
```

## Gallery
<img src="/images/smoke_plume.png" alt="Smoke plume" width="250"/>
