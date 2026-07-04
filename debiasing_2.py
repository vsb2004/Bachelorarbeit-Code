"""
Created on Tue Apr 28
@author: 
Bias Field Correction eines MRT mittels Algorithmus N4ITK und Otsu-Maske (Otsu-Thresholding zur Trennung von Vorder- und Hintergrund)
zur Entfernung niederfrequenter Intensitätsinhomogenitäten (Bias-Felder)
"""
import SimpleITK as sitk
import numpy as np

# # DICOM-Ordner einlesen
# reader = sitk.ImageSeriesReader()
# dicom_files = reader.GetGDCMSeriesFileNames("/data/pt_gr_weiskopf_7t-mri-imagedata/Phantom/Phantom_kroner/260526_143327/S4_t1_petra_tra_0.5iso_150k_5")
# reader.SetFileNames(dicom_files)
# mrt = reader.Execute()

# # ggf. Anpassung der Dimensionen (evtl. MRT durch Export als DICOM 4D, wobei 4. Dim. Singleton-Dim. ist)
# print("Originalgröße:", mrt.GetSize())
# print("Dimension:", mrt.GetDimension())
# if mrt.GetDimension() == 4:
#     mrt = mrt[:,:,:,0]
    
# print("Neue Größe:", mrt.GetSize())

# .nrrd einlesen
mrt = sitk.ReadImage("/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped.nrrd")
print("Größe:", mrt.GetSize())
print("Dimension:", mrt.GetDimension())

# Konvertierung von UInt16 (MRT) -> Float32 (N4ITK) um Algorithmus verwenden zu können
array = sitk.GetArrayFromImage(mrt).astype(np.float32)
mrt_float = sitk.GetImageFromArray(array)
mrt_float.CopyInformation(mrt)

# Erste Maske (grob)
mask_1 = sitk.OtsuThreshold(mrt_float, 0, 1, 200)
sitk.WriteImage(mask_1, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_maske_runde1.nrrd")
print("Erste Maske gespeichert")

# Erste Korrektur
corrector_1 = sitk.N4BiasFieldCorrectionImageFilter()
mrt_corrected_1 = corrector_1.Execute(mrt_float, sitk.Cast(mask_1, sitk.sitkUInt8))
log_bias_field_1 = corrector_1.GetLogBiasFieldAsImage(mrt_float)
sitk.WriteImage(log_bias_field_1, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_bias_field_runde1.nrrd")

# Zweite Maske auf vorkorrigiertem Bild
mask_2 = sitk.OtsuThreshold(mrt_corrected_1, 0, 1, 200)
sitk.WriteImage(mask_2, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_maske_runde2.nrrd")
print("Zweite Maske gespeichert")

# Finale Korrektur
corrector_2 = sitk.N4BiasFieldCorrectionImageFilter()
mrt_corrected_2 = corrector_2.Execute(mrt_float, sitk.Cast(mask_2, sitk.sitkUInt8))
log_bias_field_2 = corrector_2.GetLogBiasFieldAsImage(mrt_float)
sitk.WriteImage(log_bias_field_2, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_bias_field_runde2.nrrd")


sitk.WriteImage(mrt_corrected_2, "/data/u_benndorf_thesis/BA/Gibbs/S4_degibbs_cropped_n4_corrected.nrrd")
print("Fertig")
print("Größe:", mrt_corrected_2.GetSize())
print("Spacing:", mrt_corrected_2.GetSpacing())
