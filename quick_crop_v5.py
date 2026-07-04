#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:07:05 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# .nrrd einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/CTK_rot_reg_cropped_reg.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
del ct

#ct_cropped = arr_ct[50:750,550:1250,230:930] # Cropping 1
ct_cropped = arr_ct[160:360,:560,40:600] # Cropping 2

result = sitk.GetImageFromArray(ct_cropped)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/CTK_rot_reg_cropped_reg_cropped.nrrd")


# Konsole: plt.imshow(np.sum(arr_ct[1300:2220,700:1400,450:1100], axis=2))
