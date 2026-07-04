#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:26:56 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# .nrrd einlesen

#ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/CTK_rot_reg_cropped_reg_cropped_reg.nrrd")
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/mCTK_cropped_2.nrrd")
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/MRTK_normalized.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/bone_mask_mCT_800.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
arr_mrt = sitk.GetArrayFromImage(mrt)
arr_mask = sitk.GetArrayFromImage(mask)> 0

del ct, mrt, mask

#print(arr_mask.shape, arr_mrt.shape, arr_ct.shape)

plt.hist2d(arr_mrt[arr_mask].flatten(), arr_ct[arr_mask].flatten(), 
           bins=100, cmap="jet")
plt.colorbar()
plt.ylim(0, 1950)

plt.xlabel("MRT")
plt.ylabel("CT (HU)")
plt.show()


# # Daten aus der Maske (negative HU rausfiltern)
# valid = arr_mask & (arr_ct > 800) & (arr_mrt < 0.55)
# x = arr_mrt[valid].flatten()
# y = arr_ct[valid].flatten()

# # PCA auf 2D-Punktwolke
# data = np.column_stack([x, y])
# mean = np.mean(data, axis=0)
# data_centered = data - mean

# cov = np.cov(data_centered.T)
# eigenvalues, eigenvectors = np.linalg.eig(cov)

# # Erste Hauptkomponente (größter Eigenwert)
# principal = eigenvectors[:, np.argmax(eigenvalues)]

# # Affine Gleichung aus Hauptkomponente ableiten
# # Richtungsvektor (dx, dy) -> Steigung dy/dx
# slope = principal[1] / principal[0]
# intercept = mean[1] - slope * mean[0]

# print(f"Steigung: {slope:.2f}")
# print(f"Achsenabschnitt: {intercept:.2f}")
# print(f"Gleichung: CT = {slope:.2f} · MRT + {intercept:.2f}")

# # Plot
# plt.hist2d(x, y, bins=100, cmap="jet")
# plt.colorbar(label="Anzahl der Voxel")
# x_line = np.linspace(x.min(), x.max(), 100)
# plt.plot(x_line, slope * x_line + intercept,
#          color="white", linestyle="--", linewidth=1.5, label="PCA")
# plt.legend()
# plt.xlabel("MRT (normalisiert)")
# plt.ylabel("CT (HU)")
# plt.show()



# Daten aus der Maske (negative HU rausfiltern)
valid = arr_mask & (arr_ct > 800)
x = arr_mrt[valid].flatten()
y = arr_ct[valid].flatten()

# Lineare Regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Steigung: {slope:.2f}")
print(f"Achsenabschnitt: {intercept:.2f}")
print(f"R²: {r_value**2:.4f}")
print(f"Gleichung: CT = {slope:.2f} · MRT + {intercept:.2f}")

# Plot mit Regressionsgerade
plt.hist2d(x, y, bins=100, cmap="jet")
plt.colorbar(label="Anzahl der Voxel")
x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, slope * x_line + intercept, 
         color="white", linestyle="--", linewidth=1.5, label="lineare Regression")
plt.legend(loc = "lower left")
plt.xlabel("MRT")
plt.ylabel("CT (HU)")
plt.show()

# m, b = np.polyfit(arr_mrt[arr_mask], arr_ct[arr_mask], 1)
# x = np.linspace(arr_mrt[arr_mask].min(), arr_mrt[arr_mask].max(), 100)

# plt.scatter(arr_mrt[arr_mask], arr_ct[arr_mask], alpha=0.02, s=1, color = "teal")
# #plt.plot(x, m*x + b, "r-", label=f"CT = {m:.1f} · MRT + {b:.1f}")
# plt.xlabel("MRT")
# plt.ylabel("CT")
# plt.legend()
# plt.show()

# plt.plot(arr_mrt[arr_mask[:,:,:,2] > 1],
#          arr_ct[arr_mask[:,:,:,2] > 1], "x", alpha=0.02)