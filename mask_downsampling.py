#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:23:35 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/mCTK_cropped_2.nrrd")
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/MRTK_normalized.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/bone_mask.nrrd")

print(f"µCT Spacing: {ct.GetSpacing()}")
print(f"MRT Spacing: {mrt.GetSpacing()}")

# Auf 0.4mm downsampeln
new_spacing = [0.4, 0.4, 0.4]
new_size = [int(sz * sp / 0.4) for sz, sp in zip(mrt.GetSize(), mrt.GetSpacing())]

resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetOutputDirection(mrt.GetDirection())
resampler.SetOutputOrigin(mrt.GetOrigin())
resampler.SetInterpolator(sitk.sitkLinear)

ct_04 = resampler.Execute(ct)
mrt_04 = resampler.Execute(mrt)

# Maske: NächsterNachbar-Interpolation damit sie binär bleibt
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mask_04 = resampler.Execute(mask)

arr_ct = sitk.GetArrayFromImage(ct_04).astype(np.float32)
arr_mrt = sitk.GetArrayFromImage(mrt_04).astype(np.float32)
arr_mask = sitk.GetArrayFromImage(mask_04) > 0

del ct, mrt, mask, ct_04, mrt_04, mask_04

print(f"Grid nach Resampling: {arr_ct.shape}")

# Nur echte Knochenvoxel
valid = arr_mask & (arr_mrt > 0.2) & (arr_mrt < 0.7) & (arr_ct>0)
x = arr_mrt[valid].flatten()
y = arr_ct[valid].flatten()

print(f"Anzahl Voxel: {len(x)}")
print(f"MRT min/max/mean: {x.min():.3f} / {x.max():.3f} / {x.mean():.3f}")
print(f"CT min/max/mean: {y.min():.0f} / {y.max():.0f} / {y.mean():.0f}")

# Lineare Regression
coeffs = np.polyfit(x, y, 1)
slope, intercept = coeffs
r_squared = np.corrcoef(x, y)[0,1]**2

print(f"\nSteigung: {slope:.2f}")
print(f"Achsenabschnitt: {intercept:.2f}")
print(f"R²: {r_squared:.4f}")
print(f"Gleichung: CT = {slope:.2f} · MRT + {intercept:.2f}")

# Plot
plt.hist2d(x, y, bins=100, cmap="jet")
plt.colorbar(label="Anzahl der Voxel")
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, slope * x_line + intercept,
         color="white", linestyle="--", linewidth=1.5, label="Lineare Regression")
plt.legend()
plt.xlabel("MRT (normalisiert)")
plt.ylabel("µCT (HU)")
plt.show()