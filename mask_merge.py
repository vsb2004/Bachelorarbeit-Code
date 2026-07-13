import SimpleITK as sitk
import numpy as np


mask_img = sitk.ReadImage("/data/u_benndorf_thesis/BA/bone_mask_petra_paper.nrrd")
arr = sitk.GetArrayFromImage(mask_img)
print("Shape:", arr.shape)

# Über alle Kanäle zusammenfassen, falls 4D
if arr.ndim == 4:
    arr_combined = np.any(arr > 0, axis=-1).astype(np.uint8)
else:
    arr_combined = (arr > 0).astype(np.uint8)

print(f"Gesamt-Knochenvoxel nach Zusammenfassung: {arr_combined.sum()}")
print("Shape nach Zusammenfassung:", arr_combined.shape)


filled_img = sitk.GetImageFromArray(arr_combined.astype(np.uint8))

# Metadaten setzen
spacing = mask_img.GetSpacing()
origin = mask_img.GetOrigin()
filled_img.SetSpacing(spacing[:3] if len(spacing) == 4 else spacing)
filled_img.SetOrigin(origin[:3] if len(origin) == 4 else origin)

sitk.WriteImage(filled_img, "/data/u_benndorf_thesis/BA/bone_mask_petra_paper_final.nrrd")


