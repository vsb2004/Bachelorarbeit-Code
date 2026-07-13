
"""
Created on Mon Jun 15 14:17:49 2026

@author: Valeria Benndorf
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/402K_ROI.nrrd")
mct = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mCTK_ROI.nrrd")
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/MRTK_ROI.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/bone_mask_final.nrrd")



if mask.GetDimension() == 4:
    mask = mask[:,:,:,0]

# Spacing korrigieren (MITK hat Metadaten verloren)
ct.SetSpacing((0.125, 0.125, 0.125))
mct.SetSpacing((0.125, 0.125, 0.125))
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

resampler.SetInterpolator(sitk.sitkLinear)
mct_04 = resampler.Execute(mct)

resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mask_04 = resampler.Execute(mask)

print(f"CT Spacing nachher: {ct_04.GetSpacing()}, Size: {ct_04.GetSize()}")
print(f"MRT Spacing nachher: {mrt_04.GetSpacing()}, Size: {mrt_04.GetSize()}")
print(f"Maske Spacing nachher: {mask_04.GetSpacing()}, Size: {mask_04.GetSize()}")

# sitk.WriteImage(ct_04, "/data/u_benndorf_thesis/BA/pCT/CT_04_resampled.nrrd")
# sitk.WriteImage(mrt_04, "/data/u_benndorf_thesis/BA/pCT/MRT_04_resampled.nrrd")
# sitk.WriteImage(mct_04, "/data/u_benndorf_thesis/BA/pCT/mCT_04_resampled.nrrd")

# Histogramme
arr_ct = sitk.GetArrayFromImage(ct_04).astype(np.float32)
arr_mrt = sitk.GetArrayFromImage(mrt_04).astype(np.float32)
arr_mask = sitk.GetArrayFromImage(mask_04) > 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


if arr_mask.ndim == 4:
    arr_mask = arr_mask[..., 0]
ax1.hist(arr_mrt[arr_mask].flatten(), bins=200, alpha=0.5, color="indigo")
ax1.set_xlabel("MRT (normalisiert)")
ax1.set_ylabel("Anzahl Voxel")
ax1.set_title("MRT Histogramm (Knochenmaske)")

ax2.hist(arr_ct[arr_mask].flatten(), bins=200, alpha=0.5, color="darkred")
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
hist_ct, bins_ct = np.histogram(arr_ct[arr_mask & (arr_ct > -200)].flatten(), bins=500)
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



valid = arr_mask & (arr_mrt > 0) & (arr_mrt < 1.5) & (arr_ct > -200)
x = arr_mrt[valid]
y = arr_ct[valid]

ct_pred = slope * x + intercept

mae = np.mean(np.abs(y - ct_pred))
rmse = np.sqrt(np.mean((y - ct_pred)**2))

print(f"\n--- Fehlermetriken (Zwei-Punkt-Methode) ---")
print(f"MAE:  {mae:.2f} HU")
print(f"RMSE: {rmse:.2f} HU")


# pCT

# Umrechnung auf komplettes MRT-Array anwenden
pct_array = slope * arr_mrt + intercept

# Außerhalb der Knochenmaske: Hintergrund "Luft (-1000HU)"
pct_array_masked = np.where(arr_mask, pct_array, -1000)

print(f"pCT min/max/mean (nur Knochen): "
      f"{pct_array_masked[arr_mask].min():.0f} / "
      f"{pct_array_masked[arr_mask].max():.0f} / "
      f"{pct_array_masked[arr_mask].mean():.0f}")


# Speichern als .nrrd (mit Metadaten von mrt_04)
pct_image = sitk.GetImageFromArray(pct_array_masked.astype(np.float32))
pct_image.CopyInformation(mrt_04)  # Spacing, Origin, Direction übernehmen

# # Speichern
sitk.WriteImage(pct_image, "/data/u_benndorf_thesis/BA/pCT_seq_0.4/pCT_histogram.nrrd")


# Plots :)
# axial
z_mid = arr_mrt.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(arr_mrt[z_mid,:,:], cmap="gray") 
axes[0].set_title("MRT")
axes[1].imshow(pct_array_masked[z_mid,:,:], cmap="gray")
axes[1].set_title("pCT")
axes[2].imshow(arr_ct[z_mid,:,:], cmap="gray")
axes[2].set_title("CT")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_hist_axial.pdf")
plt.show()

# sagittal
x_mid = arr_mrt.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(arr_mrt[:, :, x_mid], cmap="gray", origin="lower")
axes[1].imshow(pct_array_masked[:, :, x_mid], cmap="gray", origin="lower")
axes[1].set_title("pCT")
axes[2].imshow(arr_ct[:, :, x_mid], cmap="gray", origin="lower")
axes[2].set_title("CT")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_hist_sagittal.pdf")
plt.show()

# coronal
y_mid = arr_mrt.shape[1] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(arr_mrt[:, y_mid, :], cmap="gray", vmin=0, vmax=1.5, origin="lower")
axes[0].set_title("MRT"),
axes[1].imshow(pct_array_masked[:, y_mid, :], cmap="gray", origin="lower")
axes[1].set_title("pCT")
axes[2].imshow(arr_ct[:, y_mid, :], cmap="gray", origin="lower")
axes[2].set_title("CT")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_hist_coronal.pdf")
plt.show()
