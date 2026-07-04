import SimpleITK as sitk

print("Starte...")
image = sitk.ReadImage("/data/hu_benndorf/mapped_MPRAGE_by_Reg #1.nii")
print("Bild geladen!")
image = sitk.Cast(image, sitk.sitkFloat32)
print("Konvertiert!")
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
print("Starte Korrektur...")
corrected = corrector.Execute(image)
print("Korrektur fertig!")
sitk.WriteImage(corrected, "/data/hu_benndorf/mprage_debiased.nii")
print("Fertig!")
