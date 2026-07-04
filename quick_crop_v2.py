#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:49:06 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# .nrrd einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v2/microCT_cropped_rot_cropped.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
del ct

ct_cropped = arr_ct[10:350,30:370,10:160]


result = sitk.GetImageFromArray(ct_cropped)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/quick_fun_abstract/v2/microCT_FBR_ROI.nrrd")
