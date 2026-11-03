# Snow mask example

[`snow-mask-alaska-west-nome.png`](snow-mask-alaska-west-nome.png) compares the unmasked true-colour image, the proposed Sentinel-2 Level-1C NDSI snow/ice mask, and Level-2A SCL class 11 as a reference. The SCL band is not used by CoastSat's Level-1C workflow. The matching Level-1C reflectance crop and SCL reference are included in [`nome-l1c-validation.npz`](nome-l1c-validation.npz), and [`render_snow_mask_example.py`](render_snow_mask_example.py) regenerates the figure.

**Source:** Sentinel-2A Level-1C scene `S2A_3WVM_20240601_0_L1C`, acquired at 22:48 UTC, with the paired Level-2A scene used only for SCL reference labels and the visual background. The cropped bands come from the public Sentinel-2 archive.

**Location:** the crop covers approximately 64.49 to 64.53 degrees north and 165.60 to 165.50 degrees west, on the Seward Peninsula shoreline of Norton Sound. It is about 5 to 10 km west of Nome, Alaska. In the Sentinel-2 image, north is at the top and the coastline runs diagonally across the frame.

With NDSI > 0.35 and green reflectance > 0.1, the mask covers 22.4% of valid pixels in this crop. SCL class 11 covers 27.2%; their intersection over union is 0.78. This is a close visual comparison on this scene, not a claim that the spectral rule reproduces SCL exactly.

The bundled scene crop is used by [`snow-mask-test.py`](../snow-mask-test.py) to check the NDSI mask against the SCL reference. The shoreline extraction is not validated by this image. Sea ice lies between the land and open water.

Source imagery: Sentinel-2A tile 3WVM from the public AWS Sentinel-2 Level-1C and Level-2A COG archives. Level-1C B3 and B11 drive the mask; Level-2A SCL and TCI are included only for validation and display.
