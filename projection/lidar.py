import torch
import os
import numpy as np
import sys
sys.path.append("..")
from kitti360scripts.helpers.project import CameraPerspective
from PIL import Image
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement


CSUPPORT = True
# Check if C-Support is available for better performance
if CSUPPORT:
    try:
        from kitti360scripts.helpers import curlVelodyneData
    except:
        CSUPPORT = False
        print('CSUPPORT is required for unwrapping the velodyne data!')
        print('Run ``CYTHONIZE_EVAL= python setup.py build_ext --inplace`` to build with cython')
        sys.exit(-1)
        
        
def loadVelodyneData(path):
    pcdFile = os.path.join(path)
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4])
    return pcd 


def readVariable(fid,name,M,N):
    # rewind
    fid.seek(0,0)
    
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break

    # return if variable identifier not found
    if success==0:
      return None
    
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert(len(line) == M*N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)

    return mat

def checkfile(filename):
    if not os.path.isfile(filename):
        raise RuntimeError('%s does not exist!' % filename)

def loadCalibrationRigid(filename):
    # check file
    checkfile(filename)

    lastrow = np.array([0,0,0,1]).reshape(1,4)
    return np.concatenate((np.loadtxt(filename).reshape(3,4), lastrow))

# Convert rotation matrix to axis angle
def Rodrigues(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = np.arctan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis * theta


def loadCalibrationCameraToPose(filename):
    # check file
    checkfile(filename)

    # open file
    fid = open(filename,'r');
     
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0,0,0,1]).reshape(1,4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
      
    # close file
    fid.close()
    return Tr


def loadVelodyneData(path):
    pcdFile = os.path.join(path)
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    pcd = np.reshape(pcd,[-1,4])
    return pcd 


class Kitti360Viewer3DRaw(object):
    def __init__(self,seq=0, mode='velodyne',kitti360_path=None):
        kitti360Path = kitti360_path
        
        if mode=='velodyne':
            self.sensor_dir='velodyne_points'
        elif mode=='sick':
            self.sensor_dir='sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

        self.kitti360Path = kitti360Path
        self.sequence = sequence
        self.loadPoses()
        self.loadExtrinsics()

    # poses are required to unwrap velodyne points to compensate for ego-motion
    def loadPoses(self):
        # load poses
        filePoses = os.path.join(self.kitti360Path, 'data_poses', self.sequence, 'poses.txt')
        poses = np.loadtxt(filePoses)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.Tr_pose_world = {}
        self.frames = frames
        for frame, pose in zip(frames, poses): 
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            self.Tr_pose_world[frame] = pose

    def loadExtrinsics(self):
        # cam_0 to velo
        fileCameraToVelo = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
        TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

        # all cameras to system center 
        fileCameraToPose = os.path.join(self.kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
        TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)
  
        self.TrVeloToPose = TrCamToPose['image_00'] @ np.linalg.inv(TrCam0ToVelo)

        # velodyne to all cameras
        self.TrVeloToCam = {}
        for k, v in TrCamToPose.items():
            # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
            TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
            TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
            # Tr(velo -> cam_k)
            self.TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd 

    def curlParameterFromPoses(self, frame):
        Tr_pose_pose = np.eye(4)

        if frame in self.Tr_pose_world.keys():
            if frame==1:
                if frame+1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame+1]) @ self.Tr_pose_world[frame]
            else:
                if frame-1 in self.Tr_pose_world.keys():
                    Tr_pose_pose = np.linalg.inv(self.Tr_pose_world[frame]) @ self.Tr_pose_world[frame-1]
        Tr_delta = np.linalg.inv(self.TrVeloToPose) @ Tr_pose_pose @ self.TrVeloToPose
        
        r = Rodrigues(Tr_delta[0:3,0:3])
        t = Tr_delta[0:3,3]
        return r.flatten(),t


    def curlVelodyneData(self, frame, pcd):
        pcd=pcd.astype(np.float64)
        pcd_curled = np.copy(pcd) 
        # get curl parameters 
        r,t = self.curlParameterFromPoses(frame)
        # unwrap points to compensate for ego motion
        pcd_curled = curlVelodyneData.cCurlVelodyneData(pcd, pcd_curled, r, t)
        return pcd_curled.astype(np.float32)