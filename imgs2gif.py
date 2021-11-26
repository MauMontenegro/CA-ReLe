import glob

from PIL import Image

# filepaths
fp_in = "Images/Emulation_*.png"
fp_out = "Images/Simulation.gif"


def images_to_gif():
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)
