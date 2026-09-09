In Data/utilities there are some adittional codes that were used to measure the SSIM and the temperature graphs.

/Data/EVAL_SESSION contains the images saved in the pi in the Operational testing session.

*refs* contains every photograph taken with its imu.json which contains the information of pitch and roll for that image, **S1_A_1.png** is the reference photo used for all SSIM comparations, and **S1_A_1.png.imu.json** for all pitch and roll comparisons.

*operador* contains the photos taken by the operador without using the device, from memory.

*ui_micro* the same but guiding the camera only with what the ui signals, starting from a small distance to force micrometry.

*ui_search* the same but starting from a large distance 2-3 metres, forcing search.