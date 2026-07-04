#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:17:49 2026

@author: benndorf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:57:23 2026

@author: benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

# Einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/CTK_rot_reg_cropped_reg_cropped_reg.nrrd")
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/MRTK_normalized.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/bone_mask_mCT_800.nrrd")

# Spacing korrigieren (MITK hat Metadaten verloren)
ct.SetSpacing((0.125, 0.125, 0.125))
mrt.SetSpacing((0.125, 0.125, 0.125))
mask.SetSpacing((0.125, 0.125, 0.125))

print(f"CT Spacing: {ct.GetSpacing()}, Size: {ct.GetSize()}")
print(f"MRT Spacing: {mrt.GetSpacing()}, Size: {mrt.GetSize()}")

# MRT auf 0.4mm resampeln → wird Referenzgrid
new_spacing = [0.4, 0.4, 0.4]
old_spacing = mrt.GetSpacing()
old_size = mrt.GetSize()
new_size = [int(sz * sp / 0.4) for sz, sp in zip(old_size, old_spacing)]

resampler_ref = sitk.ResampleImageFilter()
resampler_ref.SetOutputSpacing(new_spacing)
resampler_ref.SetSize(new_size)
resampler_ref.SetOutputDirection(mrt.GetDirection())
resampler_ref.SetOutputOrigin(mrt.GetOrigin())
resampler_ref.SetInterpolator(sitk.sitkLinear)
mrt_04 = resampler_ref.Execute(mrt)

# µCT und Maske auf exakt dieses Grid resampeln
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(mrt_04)

resampler.SetInterpolator(sitk.sitkLinear)
ct_04 = resampler.Execute(ct)

resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mask_04 = resampler.Execute(mask)

print(f"CT Spacing nachher: {ct_04.GetSpacing()}, Size: {ct_04.GetSize()}")
print(f"MRT Spacing nachher: {mrt_04.GetSpacing()}, Size: {mrt_04.GetSize()}")
print(f"Maske Spacing nachher: {mask_04.GetSpacing()}, Size: {mask_04.GetSize()}")


# Histogramme
arr_ct = sitk.GetArrayFromImage(ct_04).astype(np.float32)
arr_mrt = sitk.GetArrayFromImage(mrt_04).astype(np.float32)
arr_mask = sitk.GetArrayFromImage(mask_04) > 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(arr_mrt[arr_mask].flatten(), bins=200, color="indigo")
ax1.set_xlabel("MRT (normalisiert)")
ax1.set_ylabel("Anzahl Voxel")
ax1.set_title("MRT Histogramm (Knochenmaske)")

ax2.hist(arr_ct[arr_mask].flatten(), bins=200, color="darkred")
ax2.set_xlabel("CT (HU)")
ax2.set_ylabel("Anzahl Voxel")
ax2.set_title("CT Histogramm (Knochenmaske)")

plt.tight_layout()
plt.show()

# MRT: Spike bei 0.3 ausschließen
hist_mrt, bins_mrt = np.histogram(arr_mrt[arr_mask & (arr_mrt > 0.35)].flatten(), bins=500)
bin_centers_mrt = (bins_mrt[:-1] + bins_mrt[1:]) / 2
peak_idx_mrt = np.argmax(hist_mrt)
mrt_knochen = bin_centers_mrt[peak_idx_mrt]

# CT
hist_ct, bins_ct = np.histogram(arr_ct[arr_mask & (arr_ct > 500)].flatten(), bins=500)
bin_centers_ct = (bins_ct[:-1] + bins_ct[1:]) / 2
peak_idx_ct = np.argmax(hist_ct)
hu_knochen = bin_centers_ct[peak_idx_ct]

print(f"MRT-Knochenpeak: {mrt_knochen:.3f}")
print(f"CT-Knochenpeak: {hu_knochen:.0f} HU")


# Affine Gleichung aus zwei Punkten
mrt_wasser = 1.0
hu_wasser = 0.0

slope = (hu_knochen - hu_wasser) / (mrt_knochen - mrt_wasser)
intercept = hu_wasser - slope * mrt_wasser

print(f"\nSteigung: {slope:.2f}")
print(f"Achsenabschnitt: {intercept:.2f}")
print(f"Gleichung: CT = {slope:.2f} · MRT + {intercept:.2f}")