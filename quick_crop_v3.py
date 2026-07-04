#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:43:11 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# .nrrd einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v3/microCT_cropped_rot.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
del ct

#ct_cropped = arr_ct[1200:1700,900:1610, 250:850] # Cropping 1
ct_cropped = arr_ct[0:470,210:710,250:425] # Cropping 2

result = sitk.GetImageFromArray(ct_cropped)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/quick_fun_abstract/v3/microCT_FBR_ROI.nrrd")


# Konsole: plt.imshow(np.sum(arr_ct[1300:2220,700:1400,450:1100], axis=2))
