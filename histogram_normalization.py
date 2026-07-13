"""
Created on Tue May 26 2026

@author: Valeria Benndorf


Normalisierung einer MRT-Aufnahme eines ex-vivo Schädels (Intensität des Wasserpeaks auf 1 setzen)
zur Ermöglichung des Vergleichs von MRT-Aufnahmen mit verschiedenen Geräten/ Aufnahmezeitpunkten
und als Vorbereitung zur Anwendung der affinen Gleichung (Intensitätswert -> HU)

"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


# Bild laden
image = sitk.ReadImage("/data/u_benndorf_thesis/BA/petra_paper_K.nrrd") # auf das Micro-CT registrierte MRT (als .nrrd)
array = sitk.GetArrayFromImage(image)

# Histogramm berechnen (Anzahl der Voxel (Großteil des Hintergrunds ausgeschlossen/leere Voxel) pro Intensitätswert, 1000 Klassen)
voxels = array[array > 0]
histogram, bins = np.histogram(voxels, bins=1000)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Histogramm zeichnen
plt.figure(figsize=(10, 5))
plt.hist(voxels, bins=1000, color="indigo")
plt.xlabel("Intensität")
plt.xlim(0, 2500)
plt.ylabel("Anzahl der Voxel")
plt.title("Histogramm zur manuellen Bestimmung des Schwellenwerts")
plt.show()

# Wasserpeak finden

# # Methode 1 (manuell): Schwellenwert aus dem Histogramm oder MITK ablesen (ca. 200) 
# Schwellenwert = float(200) # Schwellenwert eingeben (Intensität zwischen Hintergrund- und Wasserpeak)

# # -> erster Peak wird ignoriert, da Hintergrundpeak (alles unter Intensität 200)
# mask_peaks = bin_centers > Schwellenwert
# peak_idx = np.argmax(histogram[mask_peaks])
# peak_value = bin_centers[mask_peaks][peak_idx]
# peak_height = histogram[mask_peaks][peak_idx]


# Methode 2 (automatisch): Otsu-Maske findet automatisch Schwellenwert zwischen Hintergrund und Gewebe
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.Execute(image)
threshold = otsu_filter.GetThreshold()
mask = array > max(threshold, 200)  # 200 als untere Absicherung gegen Artefaktvoxe
histogram_otsu, bins_otsu = np.histogram(array[mask], bins=1000)
bin_centers_otsu = (bins_otsu[:-1] + bins_otsu[1:]) / 2
peak_idx = np.argmax(histogram_otsu)
peak_value = bin_centers_otsu[peak_idx]
peak_height = histogram_otsu[peak_idx]
print(f"Schwellwert (Otsu): {threshold:.2f}")

print(f"Peak gefunden bei: {peak_value:.2f}")

# Histogramm mit markiertem Peak zeichnen und speichern

plt.figure(figsize=(10, 5))
plt.hist(voxels, bins=1000, color="indigo")
plt.xlabel("Intensität")
plt.xlim(0, 750)
plt.ylim(0, 15000000)
plt.ylabel("Anzahl der Voxel")

# Peak markieren
plt.axvline(peak_value, color="red", linestyle="--", linewidth=1.5)
plt.annotate(f"Intensität= {peak_value:.0f}",
             xy=(peak_value, peak_height),
             xytext=(peak_value + 40, peak_height),
             fontsize=14,
             color="red")

#plt.savefig("/data/u_benndorf_thesis/BA/quick_fun_abstract/v5/MRTK_histogramm.pdf")
plt.show()
#print(f"Peak gefunden bei: {peak_value:.2f}") # Methode 1
print(f"Schwellwert (Otsu): {threshold:.2f}") # Methode 2

# Normalisieren: Peak auf 1 setzen
normalized = array / peak_value
print("Peak auf 1 gesetzt")

# normalisiertes MRT als .nrrd speichern
result = sitk.GetImageFromArray(normalized)
result.CopyInformation(image)
sitk.WriteImage(result, "/data/u_benndorf_thesis/BA/petra_paper_normalized.nrrd")
print(f"Peak war: {peak_value:.2f}")
print("Fertig!")







