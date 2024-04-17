from PIL import Image, ImageSequence
import numpy as np
import sys
from scipy.ndimage import gaussian_filter
import cv2
import os
import subprocess
import re
import shutil
import tempfile
import glob
import multiprocessing

def convert_rgba_to_rgb_with_black_background(frame):
    """
    Convert an RGBA image to an RGB image, filling any transparent areas with black.
    
    Args:
        frame (PIL.Image): The source image in RGBA mode.
    
    Returns:
        PIL.Image: The converted image in RGB mode with a black background.
    """
    # Create a new black background image in RGBA mode
    black_background = Image.new('RGBA', frame.size, (0, 0, 0, 255))
    
    # Composite the RGBA frame onto the black background
    rgb_frame_with_alpha = Image.alpha_composite(black_background, frame)
    
    return rgb_frame_with_alpha

def process_gif_file(input_gif_path, output_dir="output", affix="-crt", **kwargs):
    """
    Process a GIF file by extracting its frames, applying effects to each frame with `process_image`,
    and reassembling the frames into a new GIF file.
    """
    gif = Image.open(input_gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    processed_frames = []
    for i, frame in enumerate(frames[1:]):
        # Convert RGBA to RGB with a black background if necessary
        if frame.mode == 'RGBA':
            frame = convert_rgba_to_rgb_with_black_background(frame)
        else:
            frame = frame.convert('RGB')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='w+b') as temp_frame_file:
            frame.save(temp_frame_file, format='PNG')
            temp_frame_file.flush()
            os.fsync(temp_frame_file.fileno())
            # Process the frame and save it to the output directory
            processed_frame_path = process_image(temp_frame_file.name, output_dir=output_dir, affix=affix, **kwargs)
            processed_frames.append(Image.open(processed_frame_path))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine new filename and output path
    base_name = os.path.basename(input_gif_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}.gif"
    output_gif_path = os.path.join(output_dir, new_name)

    # Save processed frames as a new GIF
    processed_frames[0].save(output_gif_path, save_all=True, append_images=processed_frames[1:], loop=0, format='GIF', duration=gif.info['duration'])

    print(f"Processed GIF saved to {output_gif_path}")

    # Clean up temporary files
    for frame in processed_frames:
        frame.close()
    for temp_frame_file in glob.glob(os.path.join(output_dir, f"*{affix}.png")):
        os.unlink(temp_frame_file)

def get_video_framerate(video_path):
    """
    Use ffprobe to get the framerate of the video.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        Framerate of the video as a float.
    """
    cmd = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', video_path]
    output = subprocess.check_output(cmd).decode().strip()
    try:
        num, den = map(int, output.split('/'))
        framerate = num / den
    except ValueError:
        framerate = 30.0  # Fallback framerate
    return framerate

def apply_opencv_sharpen(image_np, sharp_strength=1.0):
    """
    Sharpen an image using OpenCV with adjustable sharp_strength.

    Args:
        image_np: Numpy array of the image.
        sharp_strength: Multiplier for the sharpening effect. Higher values produce a stronger effect.

    Returns:
        Numpy array of the sharpened image.
    """
    # Define a basic sharpening kernel
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    
    # Scale the kernel by the sharp_strength factor
    kernel_sharpening = kernel_sharpening * sharp_strength
    
    # Ensure the kernel sums to 1 (or very close) to maintain brightness
    kernel_center = kernel_sharpening[1, 1]
    kernel_sharpening[1, 1] = kernel_center + (1 - sharp_strength) * (4 if sharp_strength > 1 else 1)

    # Apply the kernel to the image using filter2D function
    sharpened_img = cv2.filter2D(image_np, -1, kernel_sharpening)
    
    return sharpened_img

def apply_gamma_correction(image_np, gamma=2.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image_np, table)

def apply_beam_bloom(image_np, scanline_frequency=1, bloom_intensity=0.5):
    """
    Apply beam bloom effect based on the luminance of the image.
    
    Args:
        image_np: Numpy array of the image.
        scanline_frequency: The frequency of the scanlines to adjust the bloom intensity.
        bloom_intensity: Controls the intensity of the beam bloom effect.
    
    Returns:
        Numpy array of the image with beam bloom effect applied.
    """
    # Convert the image to greyscale to work with luminance
    grayscale = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])
    height = image_np.shape[0]
    for y in range(height):
        if y % scanline_frequency == 0:  # Apply effect on every nth line based on scanline frequency
            line_luminance = grayscale[y]
            # Adjust the intensity of the line based on luminance and bloom_intensity
            bloom_factor = (1 + bloom_intensity * (line_luminance / 255.0))
            # Ensure bloom_factor is applied across all RGB channels
            image_np[y] = np.clip(image_np[y] * bloom_factor[:, None], 0, 255).astype(np.uint8)
    return image_np


def apply_halation_effect(image_np, sigma_halation=5, halation_opacity=0.5):
    halation_layer = cv2.GaussianBlur(image_np, (0, 0), sigma_halation)
    blended_image = cv2.addWeighted(image_np, 1 - halation_opacity, halation_layer, halation_opacity, 0)
    return blended_image

def apply_phosphor_decay(image_np, sigma=2):
    return cv2.GaussianBlur(image_np, (0, 0), sigma)

def stretch_image(original_np, width, height):
    """
    Stretch the image to double its width and height.
    Args:
        original_np: Numpy array of the original image.
        width: Original width of the image.
        height: Original height of the image.
    Returns:
        Numpy array of the stretched image.
    """
    new_width, new_height = width * 2, height * 2
    stretched_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            original_pixel = original_np[y, x]
            stretched_image[y * 2, x * 2] = original_pixel
            stretched_image[y * 2, x * 2 + 1] = original_pixel  # Stretch horizontally
            stretched_image[y * 2 + 1, x * 2] = original_pixel  # Stretch vertically
            stretched_image[y * 2 + 1, x * 2 + 1] = original_pixel  # Stretch both directions
    return stretched_image

def apply_scanline_modulation(stretched_image, frequency=1, amplitude=0.5, phase_shift=0):
    """
    Apply scanline modulation with varying intensity using sine wave modulation for a pronounced CRT effect.
    Args:
        stretched_image: Numpy array of the stretched image.
        frequency: The frequency of the sine wave modulation for scanlines.
        amplitude: The amplitude of the sine wave modulation, affecting intensity variation.
        phase_shift: Phase shift of the sine wave to start modulation at a different point.
    Returns:
        Numpy array of the image with modulated scanlines.
    """
    height = stretched_image.shape[0]
    y_positions = np.arange(height // 2)  # Only apply to every other line to simulate scanlines
    sine_wave = (np.sin(2 * np.pi * frequency * (y_positions / (height // 2)) + phase_shift) * amplitude + 1 - amplitude) / 2
    for y in range(height // 2):
        mod_intensity = sine_wave[y % len(sine_wave)]
        stretched_image[y * 2 + 1] = (stretched_image[y * 2 + 1] * mod_intensity).astype(np.uint8)
    return stretched_image
    
def apply_saturation(image_np, saturation_factor=1.0):
    """
    Adjust the saturation of an image using OpenCV.

    Args:
        image_np: Numpy array of the image.
        saturation_factor: The factor to adjust saturation by. 1.0 keeps the saturation unchanged,
                           values below 1.0 decrease saturation, values above 1.0 increase saturation.

    Returns:
        Numpy array of the image with adjusted saturation.
    """
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv_image)

    # Adjust the saturation channel by the saturation_factor
    s = cv2.multiply(s, saturation_factor)

    # Merge the adjusted saturation channel back with the original hue and value channels
    adjusted_hsv = cv2.merge((h, s, v))

    # Convert the image back to BGR color space
    adjusted_bgr = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_bgr
    
    
    

def process_image(image_path, output_dir="output", affix="-crt", **kwargs):
    """
    Process the image by stretching and applying scanline modulation.
    Args:
        image_path: Path to the input image.
        output_dir: Directory where the processed image will be saved.
        affix: String to be affixed to the output filename.
        **kwargs: Additional keyword arguments for image processing effects.
    Returns:
        Path to the processed image.
    """
    original_np = np.array(Image.open(image_path))
    height, width = original_np.shape[:2]

    stretched_image = cv2.resize(original_np, (width * 2, height * 2), interpolation=cv2.INTER_NEAREST)
    blurred_image = apply_phosphor_decay(stretched_image, sigma=kwargs.get('sigma', 2))
    halation_image = apply_halation_effect(blurred_image, sigma_halation=kwargs.get('sigma_halation', 5), halation_opacity=kwargs.get('halation_opacity', 0.5))
    final_image = apply_scanline_modulation(halation_image, kwargs.get('frequency', 1), kwargs.get('amplitude', 0.5), kwargs.get('phase_shift', 0))
    final_image_with_bloom = apply_beam_bloom(final_image, scanline_frequency=kwargs.get('scanline_frequency', 2), bloom_intensity=kwargs.get('bloom_intensity', 0.5))
    sharpened_image = apply_opencv_sharpen(final_image_with_bloom, sharp_strength=kwargs.get('sharp_strength', 0.1))
    saturated_image = apply_saturation(sharpened_image, saturation_factor=kwargs.get('saturation_factor', 1.1))
    gamma_corrected_image = apply_gamma_correction(saturated_image, gamma=kwargs.get('gamma', 2.2))

    # Determine new filename and output path
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}{affix}{ext}"
    output_path = os.path.join(output_dir, new_name)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Convert the color space from BGR to RGB
    gamma_corrected_image_rgb = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, gamma_corrected_image_rgb)
    print(f"Processed image saved to {output_path}")
    
    return output_path

def process_video_file(input_video_path, output_video_filename, **kwargs):
    """
    Process a video file by extracting its frames, applying effects to each frame,
    and reassembling the frames into a new video file using the original video's framerate.
    
    Args:
        input_video_path: Path to the input video file.
        output_video_path: Path where the processed video will be saved.
        **kwargs: Additional keyword arguments passed to the image processing function.
    """
    temp_frame_dir = "imgs"  # Use the hardcoded directory from extract.py
    os.makedirs(temp_frame_dir, exist_ok=True)
    
    current_dir = os.getcwd()  # Get the current working directory
    output_video_path = os.path.join(current_dir, output_video_filename)

    # Extract frames using external script
    subprocess.call(['python', 'extract.py', input_video_path])

    # Get the list of extracted frame files
    frame_files = sorted(os.listdir(temp_frame_dir))

    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() // 2))
    print(f"Processing frames with {pool._processes} worker process(es)...")
    
    processed_frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frame_dir, frame_file)
        if frame_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            processed_frame = pool.apply_async(process_image, (frame_path,), {'output_dir': 'output', **kwargs})
            processed_frames.append(processed_frame)

    # Wait for all frames to be processed
    for processed_frame in processed_frames:
        processed_frame.wait()

    pool.close()
    pool.join()
    
    # Get the original video's framerate
    framerate = get_video_framerate(input_video_path)

    # Reassemble video from processed frames in the "output" directory
    processed_frame_dir = "output"  # Directory where processed frames are saved
    # Assuming processed frames are saved with filenames like "0000001-crt.png", adjust the pattern accordingly
    ffmpeg_command_assemble = [
        'ffmpeg', '-framerate', str(framerate), '-i', os.path.join(processed_frame_dir, '%07d-crt.png'),
        '-pix_fmt', 'yuv420p', '-c:v', 'av1_nvenc', '-b:v', '10M', output_video_path.replace('.mkv', '_no_audio.mkv')
    ]
    subprocess.call(ffmpeg_command_assemble)
    
    # Add audio back to the processed video
    ffmpeg_command_audio = [
        'ffmpeg', '-i', output_video_path.replace('.mkv', '_no_audio.mkv'),
        '-i', input_video_path,
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', output_video_path
    ]
    subprocess.call(ffmpeg_command_audio)

    # Cleanup temporary directories and intermediate no-audio video
    dirs_to_delete = [temp_frame_dir, processed_frame_dir]
    file_to_delete = output_video_path.replace('.mkv', '_no_audio.mkv')

    for dir_path in dirs_to_delete:
        shutil.rmtree(dir_path, ignore_errors=True)
        print(f"Deleted directory: {dir_path}")

    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)
        print(f"Deleted file: {file_to_delete}")

def process_input(input_path, **kwargs):
    """
    Determine if input is an image or video and process accordingly.
    """
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        process_image(input_path, **kwargs)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Extract directory, base filename, and extension
        dir_name = os.path.dirname(input_path)
        base_name, ext = os.path.splitext(os.path.basename(input_path))
        # Construct new output path with '_processed' suffix before the extension
        output_video_path = os.path.join(dir_name, f"{base_name}_processed{ext}")
        process_video_file(input_path, output_video_path, **kwargs)
    elif input_path.lower().endswith(('.gif')):
        process_gif_file(input_path, **kwargs)
    else:
        print("Unsupported file format.")

if __name__ == "__main__":
    # The existing command-line interface remains unchanged
    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_input")
    else:
        input_path = sys.argv[1]
        process_input(input_path, 
                      frequency=0.1,  # Controls the frequency of the sine wave used in scanline modulation. Lower values result in fewer, more spaced-out scanlines.
                      amplitude=0.25,  # Determines the amplitude of the sine wave for scanline modulation, affecting the intensity variation of the scanlines.
                      phase_shift=1.0,  # Adjusts the phase of the sine wave used in scanline modulation, shifting the starting point of the wave pattern.
                      sigma=0.6,  # Blur -- Standard deviation for the Gaussian blur in the phosphor decay effect, controlling the amount of blur applied.
                      sigma_halation=2.5,  # Standard deviation for the Gaussian blur in the halation effect, dictating the blur intensity around bright areas.
                      halation_opacity=0.5,  # Opacity factor for blending the halation effect with the original image, controlling the intensity of the halation effect.
                      scanline_frequency=1.0,  # Determines the frequency of the scanlines for the beam bloom effect, adjusting how often the effect is applied vertically.
                      bloom_intensity=1.2,  # Controls the intensity of the beam bloom effect, higher values increase the glow around bright image areas.
                      gamma=1.1,  # The gamma correction value, lower than 1 darkens the image, and higher than 1 brightens it.
                      sharp_strength=0.05,
                      saturation_factor=1.15)  # Multiplier for the sharpening effect strength, higher values produce a stronger sharpening effect.
