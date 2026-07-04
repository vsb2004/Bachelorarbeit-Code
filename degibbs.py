"""
Created on Tue Jun 16 09:30:27 2026
@author: Valeria Benndorf
"""
import matplotlib.pyplot as plt
import SimpleITK as sitk
from dipy.denoise.gibbs import gibbs_removal
import numpy as np

# DICOM-Ordner einlesen
reader = sitk.ImageSeriesReader()
dicom_files = reader.GetGDCMSeriesFileNames("/data/pt_gr_weiskopf_7t-mri-imagedata/Phantom/Phantom_kroner/260526_143327/S4_t1_petra_tra_0.5iso_150k_5")
reader.SetFileNames(dicom_files)
mrt = reader.Execute()
print("Originalgröße:", mrt.GetSize())
print("Dimension:", mrt.GetDimension())
if mrt.GetDimension() == 4:
    mrt = mrt[:, :, :, 0]
    
# # .nrrd einlesen
# mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/MRT_to_mCT/Reg_S4/S4_cropped.nrrd")
# print("Größe:", mrt.GetSize())
# print("Dimension:", mrt.GetDimension())

mrt_array = sitk.GetArrayFromImage(mrt)
print("Array-Shape:", mrt_array.shape)
print("Datentyp original:", mrt_array.dtype)

# Schnitt entlang Achse 0
slice0 = mrt_array[mrt_array.shape[0] // 2, :, :]
# Schnitt entlang Achse 1
slice1 = mrt_array[:, mrt_array.shape[1] // 2, :]
# Schnitt entlang Achse 2
slice2 = mrt_array[:, :, mrt_array.shape[2] // 2]

# Plot originale Aufnahme
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sl, title in zip(axes, [slice0, slice1, slice2], ["Achse 0", "Achse 1", "Achse 2"]):
    vmin, vmax = np.percentile(sl, [1, 99])
    ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)
fig.savefig("/data/u_benndorf_thesis/BA/Gibbs/Plot_axis_S4.pdf")
plt.show()

# vor der Korrektur in float32 konvertieren, sonst Integer-Überlauf
# bei negativen Subvoxel-Überschwingern des FFT-basierten Algorithmus
orig = mrt_array.astype(np.float32)

step0 = gibbs_removal(orig, slice_axis=0, inplace=False)
step1 = gibbs_removal(step0, slice_axis=1, inplace=False)
step2 = gibbs_removal(step1, slice_axis=2, inplace=False)

diff_0 = orig - step0
diff_1 = step0 - step1
diff_2 = step1 - step2

print("Max. Änderung Achse 0:", np.abs(diff_0).max())
print("Max. Änderung Achse 1:", np.abs(diff_1).max())
print("Max. Änderung Achse 2:", np.abs(diff_2).max())

# Vor dem Speichern zurück in den ursprünglichen Datentyp (uint16),
# mit Clipping statt direktem Cast, um erneuten Überlauf zu vermeiden
step2_clipped = np.clip(np.round(step2), 0, 65535).astype(np.uint16)

step2_img = sitk.GetImageFromArray(step2_clipped)
step2_img.CopyInformation(mrt)
sitk.WriteImage(step2_img, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs.nrrd")  # zur Weiterverarbeitung in MITK
sitk.WriteImage(step2_img, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs.nii.gz")  # zur Analyse in ImageJ

# Korrigierte Slices (aus dem float-Array step2, vor dem uint16-Clipping,
slice0_corr = step2[step2.shape[0] // 2, :, :]
slice1_corr = step2[:, step2.shape[1] // 2, :]
slice2_corr = step2[:, :, step2.shape[2] // 2]

# Plot korrigierte Aufnahme
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, sl, sl_orig, title in zip(
    axes,
    [slice0_corr, slice1_corr, slice2_corr],
    [slice0, slice1, slice2],          # Original-Slices für die Skalierung
    ["Achse 0", "Achse 1", "Achse 2"],
):
    vmin, vmax = np.percentile(sl_orig, [1, 99])   # aus Original berechnet
    ax.imshow(sl, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title)
fig.savefig("/data/u_benndorf_thesis/BA/Gibbs/Plot_axis_S4_degibbs.pdf")
plt.show()


diff = orig - step2  

idx_0 = orig.shape[0] // 2
diff_slice_0 = diff[idx_0, :, :]

idx_1 = orig.shape[1] // 2
diff_slice_1 = diff[:, idx_1, :]

idx_2 = orig.shape[2] // 2
diff_slice_2 = diff[:, :, idx_2]

# einheitliche, symmetrische Farbskala über alle drei Slices hinweg
vmax = np.percentile(
    np.abs(np.concatenate([diff_slice_0.ravel(), diff_slice_1.ravel(), diff_slice_2.ravel()])),
    99
)

# Plot Differenz
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].imshow(diff_slice_0, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[0].set_title("Achse 0")
axes[1].imshow(diff_slice_1, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[1].set_title("Achse 1")
im2 = axes[2].imshow(diff_slice_2, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[2].set_title("Achse 2")
fig.colorbar(im2, ax=axes, shrink=0.8, label='Original - Degibbs', fraction=0.2, pad=0.015)
fig.savefig("/data/u_benndorf_thesis/BA/Gibbs/Plot_axis_original_degibbs.pdf")
plt.show()


# Beispiel: Ausschnitt um die Kontur in Achse 1
y0, y1 = 20, 100
x0, x1 = 150, 280

crop_orig = slice1[y0:y1, x0:x1]
crop_corr = slice1_corr[y0:y1, x0:x1]

vmin, vmax = np.percentile(crop_orig, [1, 99])

# Plot Ausschnitt
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(crop_orig, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title("Original (Ausschnitt)")
axes[1].imshow(crop_corr, cmap='gray', vmin=vmin, vmax=vmax)
axes[1].set_title("Degibbst (Ausschnitt)")
fig.savefig("/data/u_benndorf_thesis/BA/Gibbs/Plot_axis_original_degibbs_Ausschnitt.pdf")
plt.show()