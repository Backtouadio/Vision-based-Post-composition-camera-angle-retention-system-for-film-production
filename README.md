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

# Documentation of each files function:

# On the Device (Raspberry Pi)
nucleo.py: It holds all the shared definitions and adjustable settings in one place. It doesn't perform actions itself, it ensures the rest of the code uses the same configuration.

camara.py: Controls the camera. It locks the focus, brightness, and color settings so the live video and your saved reference photos are always perfectly consistent and easy to compare.

imu.py: Reads the motion sensors to give you two numbers exactly how much is the camera roll and pitch.

vision.py: The core of the system. It compares the live camera view to your reference photo and calculates how far off you are. It automatically shifts between a broad search when you're far away and precise tracking once you get close.

guia.py: Translates the raw math into human readable instructions (like "move right," "tilt down," or "turn left"). It also smooths out the data so the guiding arrows don't jump around from small hand movements.**Suavizado is False, meaning suavizado is off**

ui_local.py: Shows the guiding arrows directly on the device's screen. If you are connected via a text-only terminal, it prints the instructions line by line instead.

red.py: Handles all communication. It sends the captured data to your laptop, runs a simple web page where you can manage your reference photos.

main.py: Manages the main loop, and makes sure all the different parts hand information to each other in the right order.

# On the Laptop (PC)
puente.py: Receives updates from the device over the network and feeds them into your main visual display in the web browser (operator_ui_web.html).
