# Vision-based-Post-composition-camera-angle-retention-system-for-film-production
The complete source code developed for this thesis including the computer vision pipeline, the hardware telemetry integration, and the web-based user interface.
The repository also contains the 3D CAD models (STL files) designed for the custom device enclosure. 

September 2026, author: Jin Weijian Victor Chen

Code folder has two subdirectories, one for the files needed in the pi, and the ones that run in the laptop.

To run the pipeline:

Laptop --> Folder pc : **python3 puente.py --live --udp-port 5005**

Pi --> Folder pi: **python3 main.py --setup --headless --telemetria 192.168.100.166:5005** 

Visit web**http://localhost:8080** to see the UI.

**Ip adresses and ports need to be adapted in order to function.
