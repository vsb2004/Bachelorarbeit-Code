"""
@author: Valeria Benndorf

Vergrößerung des Volumens des mCT um einen Rand (Padding), um den Verlust von
Bildinformationen bei der späteren manuellen Vorausrichtung zu vermeiden
"""


import SimpleITK as sitk
import os

input_dir = "/data/u_benndorf_thesis/BA"
output_dir = "/data/u_benndorf_thesis/BA"
microct_path = os.path.join(input_dir, "halle_skull.nrrd")

microct = sitk.ReadImage(microct_path)

# 200 Voxel = 200 × 0.125mm = 25mm 
padded = sitk.ConstantPad(
    microct,
    padLowerBound=[200, 200, 200],
    padUpperBound=[200, 200, 200],
    constant=-1000                  # entspricht CT-Zahl von Luft in HU
)

# Speichern
sitk.WriteImage(padded, os.path.join(output_dir, "mct_padded.nii"))
print("Originalgröße:  ", microct.GetSize())
print("Neue Größe:     ", padded.GetSize())