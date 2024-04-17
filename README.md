# mediaToScanline
Covert images, gif, and video to have a scanline effect

# Showcase
![Alt Text](tumbl.png)
![Alt Text](output/tumbl-crt.png)

![Alt Text](campfire.gif)
![Alt Text](output/campfire-crt.gif)

![Alt Text](fighting-dragon.jpg)
![Alt Text](output/fighting-dragon-crt.jpg)


# Video?
Works with videos too. It works by first extracting the frames, then converting them via concurrency, spawning about 1/2 of CPU cores to process the images.
Once that is done, it uses the new av1 (requires the new nvidia gpus) and puts the video back together along with the right framerate and sound (no audio quality loss, it's not re-encoded).

If your GPU doesn't support the new av1 encoder, you can change it to use e.g. 265, or just change the quality value in the scanline.py, search for "ffmpeg" and you should find it there.

# Adjust settings?
At the very bottom of the scanline.py, you'll find the various values used. I made those based on my assumption of how a CRT might look like. It might be too blurry for some, or not blurry enough for others, so you're welcome to adjust it to fit your needs.

# Installation
pip install -r req.txt

make sure ffmpeg is installed for video stuff

then you can run something like:
python scanline.py ./tumbl.png
