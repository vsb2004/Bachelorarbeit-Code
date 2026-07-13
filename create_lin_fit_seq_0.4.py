"""
Created on Jul 10 2026

@author: Valeria Benndorf

Downsampeln der Datensaätze und Ermittlung einer affinen Gleichung (Intensitätswert -> HU)
zur Erstellung eines MRT-basierten pseudo-CTs
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Einlesen
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/402K_ROI.nrrd")
mct = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mCTK_ROI.nrrd")
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/MRTK_ROI.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/pCT_seq_0.4/bone_mask_final.nrrd")

# ggf. 4. Dim entfernen
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

resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetOutputDirection(mrt.GetDirection())
resampler.SetOutputOrigin(mrt.GetOrigin())
resampler.SetInterpolator(sitk.sitkLinear)

ct_04 = resampler.Execute(ct)
mct_04 = resampler.Execute(mct)
mrt_04 = resampler.Execute(mrt)

# Maske: Nächster-Nachbar-Interpolation, damit sie binär bleibt
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mask_04 = resampler.Execute(mask)

arr_ct = sitk.GetArrayFromImage(ct_04).astype(np.float32)
arr_mct = sitk.GetArrayFromImage(mct_04).astype(np.float32)
arr_mrt = sitk.GetArrayFromImage(mrt_04).astype(np.float32)
arr_mask = sitk.GetArrayFromImage(mask_04) > 0

if arr_mask.ndim == 4:
    arr_mask = arr_mask[:, :, :, 0]

print(f"Grid nach Resampling: {arr_ct.shape}")

# Nur echte Knochenvoxel
valid = arr_mask & (arr_mrt > 0) & (arr_mrt < 1.5) & (arr_ct>0)
x = arr_mrt[valid].flatten()
y = arr_ct[valid].flatten()

# Voxelanzahl, Min, Mittelwert, Max von MRT und CT
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
plt.hist2d(x, y, bins=100, cmap="magma")
# Colorbar anpassen
cbar = plt.colorbar(label="Anzahl der Voxel")
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0)) 
cbar.ax.yaxis.set_major_formatter(formatter)
offset_text = cbar.ax.yaxis.get_offset_text()
offset_text.set_x(3.5)  

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, slope * x_line + intercept,
         color="white", linestyle="--", linewidth=1.5, label = f"CT = {slope:.3f} · MRT + {intercept:.3f}")
plt.legend(loc='lower left', labelcolor='white', frameon=False, fontsize=11)
plt.xlabel("MRT (normalisiert)")
plt.ylabel("CT [HU]")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/lin_plot_sequenz_0.4.pdf", dpi=600)
plt.show()

# Plot Poster
plt.hist2d(x, y, bins=100, cmap="magma")
# Colorbar anpassen
cbar = plt.colorbar(label="Voxel count")
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0)) 
cbar.ax.yaxis.set_major_formatter(formatter)
offset_text = cbar.ax.yaxis.get_offset_text()
offset_text.set_x(3.5)  

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x_line, slope * x_line + intercept,
         color="white", linestyle="--", linewidth=1.5, label = f"CT = {slope:.3f} · MRI + {intercept:.3f}")
plt.legend(loc='lower left', labelcolor='white', frameon=False, fontsize=11)
plt.xlabel("MRI (normalized)")
plt.ylabel("CT [HU]")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/lin_plot_poster.png", dpi=600)
plt.show()


# Umrechnung auf komplettes MRT-Array anwenden
pct_array = slope * arr_mrt + intercept

# Außerhalb der Knochenmaske: Hintergrund: Luft (-1000 HU)
pct_array_masked = np.where(arr_mask, pct_array, -1000)

print(f"pCT min/max/mean (nur Knochen): "
      f"{pct_array_masked[arr_mask].min():.0f} / "
      f"{pct_array_masked[arr_mask].max():.0f} / "
      f"{pct_array_masked[arr_mask].mean():.0f}")


# Speichern als .nrrd (mit Metadaten von mrt_04)
pct_image = sitk.GetImageFromArray(pct_array_masked.astype(np.float32))
pct_image.CopyInformation(mrt_04)  # Spacing, Origin, Direction übernehmen
sitk.WriteImage(pct_image, "/data/u_benndorf_thesis/BA/pCT_seq_0.4/pCT_lin.nrrd")


# Plot :)
# axial
z_mid = arr_mrt.shape[0] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(arr_mrt[z_mid,:,:], cmap="gray", vmin=0, vmax=1.5) # Werte > 1.5 sind Registrierungsartefakte
axes[0].set_title("MRT")
axes[1].imshow(pct_array_masked[z_mid,:,:], cmap="gray")
axes[1].set_title("pCT")
axes[2].imshow(arr_ct[z_mid,:,:], cmap="gray")
axes[2].set_title("CT")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_lin_axial.pdf")
plt.show()

# sagittal
x_mid = arr_mrt.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(arr_mrt[:, :, x_mid], cmap="gray", vmin=0, vmax=1.5, origin="lower")
axes[0].set_title("MRT"),
axes[1].imshow(pct_array_masked[:, :, x_mid], cmap="gray", origin="lower")
axes[1].set_title("pCT")
axes[2].imshow(arr_ct[:, :, x_mid], cmap="gray", origin="lower")
axes[2].set_title("CT")
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_lin_sagittal.pdf")
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
plt.savefig("/data/u_benndorf_thesis/BA/pCT_seq_0.4/mrt_pct_ct_plot_lin_coronal.pdf")
plt.show()

# plots poster
def add_scalebar(ax, spacing_mm=0.75, length_cm=1, img_shape=None):
    length_mm = length_cm * 10
    length_px = length_mm / spacing_mm
    
    if img_shape is not None:
        h, w = img_shape
        x0 = w * 0.8   
        y0 = h * 0.057   
    else:
        x0, y0 = 20, 20
    
    ax.plot([x0, x0 + length_px], [y0, y0], color="black", lw=3)
    ax.text(x0 + length_px / 2, y0 + h*0.02, f"{length_cm} cm",
             color="black", fontsize=12, ha="center", va="bottom")

crop = 1  # Anzahl Voxel, die am Rand weggeschnitten werden

# MRI
y_mid = arr_mrt.shape[1] // 2
img_shape = arr_mrt[:, y_mid, :].shape

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(arr_mrt[crop:-crop, y_mid, crop:-crop], cmap="gray", vmin=0, vmax=1.5, origin="lower")
ax.set_title("MRI (0.4mm)")
add_scalebar(ax, spacing_mm=0.75, length_cm=1, img_shape=img_shape)
ax.axis("off")
plt.savefig("/data/u_benndorf_thesis/BA/MRI_poster.png", dpi=600, bbox_inches="tight")
plt.show()



# # CT  
# y_mid = arr_mrt.shape[1] // 2
# img_shape = arr_mrt[:, y_mid, :].shape

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(arr_ct[crop:-crop, y_mid, crop:-crop], cmap="gray",origin="lower")
# ax.set_title("CT")
# add_scalebar(ax, spacing_mm=0.4, length_cm=1, img_shape=img_shape)
# ax.axis("off")
# #plt.savefig("/data/u_benndorf_thesis/BA/CT_poster.png", dpi=600)
# plt.show()

# # mCT
# y_mid = arr_mrt.shape[1] // 2
# img_shape = arr_mrt[:, y_mid, :].shape

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(arr_mct[crop:-crop, y_mid, crop:-crop], cmap="gray",origin="lower")
# ax.set_title("µCT")
# add_scalebar(ax, spacing_mm=0.4, length_cm=1, img_shape=img_shape)
# ax.axis("off")
# #plt.savefig("/data/u_benndorf_thesis/BA/mCT_poster.png", dpi=600)
# plt.show()

# # pCT
# y_mid = arr_mrt.shape[1] // 2
# img_shape = arr_mrt[:, y_mid, :].shape

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(arr_mct[crop:-crop, y_mid, crop:-crop], cmap="gray",origin="lower")
# ax.set_title("pCT")
# add_scalebar(ax, spacing_mm=0.4, length_cm=1, img_shape=img_shape)
# ax.axis("off")
# #plt.savefig("/data/u_benndorf_thesis/BA/pCT_poster.png", dpi=600)
# plt.show()