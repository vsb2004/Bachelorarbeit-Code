#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:33:48 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# .nrrd einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/reg_norm_mrt.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
del ct

ct_cropped = arr_ct[1220:1600,1000:1720, 250:750]

result = sitk.GetImageFromArray(ct_cropped)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/quick_fun_abstract/v3/MRT_cropped.nrrd")

# Konsole: plt.imshow(np.sum(arr_ct[1300:2220,700:1400,450:1100], axis=2))



