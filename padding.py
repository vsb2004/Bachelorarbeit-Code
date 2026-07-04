import SimpleITK as sitk

microct = sitk.ReadImage("/data/tu_benndorf/Halle_skull/halle_skull.nrrd")

# Kleineres Padding nötig als beim CT,
# da Micro-CT Voxel sehr klein sind (0.125mm)
# 200 Voxel = 200 × 0.125mm = 25mm Puffer
padded = sitk.ConstantPad(
    microct,
    padLowerBound=[200, 200, 200],
    padUpperBound=[200, 200, 200],
    constant=-1000
)

sitk.WriteImage(padded, "/data/tu_benndorf_private/microct_padded.nrrd")
print("Originalgröße:  ", microct.GetSize())
print("Neue Größe:     ", padded.GetSize())