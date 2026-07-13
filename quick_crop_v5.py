"""
Created on Sun Jun 14 13:07:05 2026

@author: Valeria Benndorf
"""

import SimpleITK as sitk

# import .nrrd
ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/petra_paper_K_cropped_reg.nrrd")
#ct = sitk.ReadImage("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/mCTK_cropped.nrrd")

arr_ct = sitk.GetArrayFromImage(ct)
del ct

#ct_cropped = arr_ct[50:750,550:1250,230:930] # Cropping 1
ct_cropped = arr_ct[160:360,:560,40:600] # Cropping 2

result = sitk.GetImageFromArray(ct_cropped)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/petra_paper_K_cropped_reg_cropped.nrrd")


# console: plt.imshow(np.sum(arr_ct[:,:,:], axis=2))
#plt.imshow(np.sum(arr_ct[50:750,550:1250,230:930], axis=2))