"""
Created on Wed Jun  3 22:54:19 2026

@author: benndorf
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# # Bilder laden
# reader = sitk.ImageSeriesReader()
# dicom_files = reader.GetGDCMSeriesFileNames("/data/pt_gr_weiskopf_7t-mri-imagedata/Phantom/Phantom_kroner/260526_143327/S4_t1_petra_tra_0.5iso_150k_5")
# reader.SetFileNames(dicom_files)
# original = reader.Execute()

# .nrrd einlesen
original = sitk.ReadImage("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped.nrrd")
print("Größe:", original.GetSize())
print("Dimension:", original.GetDimension())

if original.GetDimension() == 4:
    original = original[:,:,:,0]

corrected = sitk.ReadImage("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_n4_corrected.nrrd")
mask = sitk.ReadImage("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_maske_runde2.nrrd")
bias_field = sitk.ReadImage("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_bias_field_runde2.nrrd")

arr_orig = sitk.GetArrayFromImage(original)
arr_corr = sitk.GetArrayFromImage(corrected)
arr_mask = sitk.GetArrayFromImage(mask)
arr_field = sitk.GetArrayFromImage(bias_field)

# Sagittal-Slice (wie in deinem Bild)
z_slice = arr_orig.shape[0] // 2 # Mittlerer Slice

orig_slice = arr_orig[z_slice]
corr_slice = arr_corr[z_slice]
mask_slice = arr_mask[z_slice]
bias_field_slice = arr_field[z_slice]

# Differenz berechnen — zwei Optionen:
# absolute Differenz
diff = np.abs(orig_slice - corr_slice)

# Visualisierung
fig, axes = plt.subplots(1, 5, figsize=(14, 10), facecolor='black')
plt.subplots_adjust(left=0.01, right=0.88, top=0.92, bottom=0.02, wspace=0.05)

axes[0].imshow(orig_slice, cmap='gray')
axes[0].set_title('MRT (original)', color='white', fontsize=12)
axes[0].axis('off')

axes[1].imshow(bias_field_slice, cmap='gray')
axes[1].set_title('Bias Field', color='white', fontsize=12)
axes[1].axis('off')

axes[2].imshow(mask_slice, cmap='gray')
axes[2].set_title('Maske', color='white', fontsize=12)
axes[2].axis('off')

axes[3].imshow(corr_slice, cmap='gray')
axes[3].set_title('MRT (debiased)', color='white', fontsize=12)
axes[3].axis('off')

# Differenz mit Colormap
im = axes[4].imshow(diff, cmap='magma')  # oder: 'hot', 'magma', 'plasma'
axes[4].set_title('Differenz', color='white', fontsize=12)
axes[4].axis('off')

# Colorbar in eigene Axes rechts außen
cbar_ax = fig.add_axes([0.9, 0.33, 0.015, 0.3])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Intensitätsdifferenz', color='white', fontsize=12)
cbar.ax.tick_params(colors='white', labelsize=11)
cbar.ax.yaxis.label.set_color('white')

fig.savefig("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_n4_plot.pdf", bbox_inches='tight', facecolor='black')
plt.show()

