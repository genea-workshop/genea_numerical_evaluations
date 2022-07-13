# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *

import joblib as jl
import glob

# 18 joints (only upper body)
#target_joints = ['b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

# 50 joints (upper body with fingers)
target_joints = ['p_r_scap', 'p_l_scap', 'body_world', 'b_root', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

# 24 joints (upper and lower body excluding fingers)
# target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_neck0', 'b_head']

#56 joints (upper and lower body including fingers)
#target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']


def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, fps):
    p = BVHParser()

    data_all = list()
    for f in files:
        ff = f
        print(ff)
        data_all.append(p.parse(ff))

    data_pipe = Pipeline([
       ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
       #('root', RootNormalizer()),
       ('jtsel', JointSelector(target_joints, include_root=False)),
       ('exp', MocapParameterizer('position')),
       ('np', Numpyfier())
    ])


    out_data = data_pipe.fit_transform(data_all)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == len(files)
    
    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav'))
        
    fi=0
    for f in files:
        ff = f.split("/")[-1]
        print(ff)
        np.save(os.path.join(dest_dir, ff[:-4] + ".npy"), out_data[fi])
        fi=fi+1



if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--bvh_dir', '-orig', required=True,
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', required=True,
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    files = sorted([f for f in glob.iglob(params.bvh_dir+'/*.bvh')])

    extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir , fps=30)
