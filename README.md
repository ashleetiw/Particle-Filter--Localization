# Particle Filter based localization on UTIAS Multi-Robot Cooperative Localization and Mapping dataset

## Dataset
UTIAS Multi-Robot Cooperative Localization and Mapping is 2D indoor feature-based dataset. For details
please refer here. This project contains Dataset0 (MRSLAM Dataset4, Robot3) and Dataset1 (MRCLAM
Dataset9, Robot3). All algorithms are using Dataset1 to generate the following results.
Each dataset contains five files:
1. Odometry.dat: Control data (translation and rotation velocity)
2. Measurement.dat: Measurement data (range and bearing data for visually observed landmarks and other
robots)
3. Groundtruth.dat: Ground truth robot position (measured via Vicon motion capture – use for assessment
only)
4. LandmarkGroundtruth.dat: Ground truth landmark positions (measured via Vicon motion capture)
5. Barcodes.dat: Associates the barcode IDs with landmark IDs.

The data is processed in the following way:
1. Use all Odometry data
2. Only use Measurement data for landmark 6 to 20 (1 to 5 are other robots)
3. Use Groundtruth data to plot the robot state ground truth
4. Use Landmark Groundtruth only for localization problem
5. Associate Landmark Groundtruth with Barcodes to get landmark index from measurement
6. Combine Odometry data with Measurement data ans sort by timestamp as input data
7. The error because of collecting the odometry data

### Localization using only motion model
![m](https://github.com/ashleetiw/Particle-Filter--Localization/blob/master/images/pf4.png)
### Particle filter implementation on the dataset
![PF](https://github.com/ashleetiw/Particle-Filter--Localization/blob/master/images/video.gif)

### Run 
python `code.py`

