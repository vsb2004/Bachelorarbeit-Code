#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 08:45:16 2026

@author: benndorf
"""

import SimpleITK as sitk

# DICOM-Ordner einlesen
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames("/data/pt_gr_weiskopf_7t-mri-imagedata/Phantom/Phantom_kroner/260602_130520/S4_t1_petra_tra_0.4iso_preScan_noFilter_150k")
reader.SetFileNames(dicom_files)
mrt = reader.Execute()

sitk.WriteImage(mrt, "/data/u_benndorf_thesis/BA/Kalotte.nii.gz")  # oder .tif