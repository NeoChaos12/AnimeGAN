import os.path
from os import makedirs
import subprocess
import argparse
import cv2 as cv
import numpy as np
import logging
from glob import glob
from math import ceil, floor

def parse_args():

    desc = "Script for tiling a set of large images, calling AnimeGAN on the tiled images, and then stitching them " \
           "back together."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/haoyao_style_gan',
                        help='Directory name to load the checkpoint from.')
    parser.add_argument('--input_dir', type=str, default='dataset/navdeep_wedding',
                        help='Directory name to load the input images from.')
    parser.add_argument('--output_dir', type=str, default='result/navdeep_wedding',
                        help='Directory name to save the generated images into.')
    parser.add_argument('--type', type=str, default='jpg',
                        help='File extension of the images.')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for input images.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enable debug mode logging.')

    return parser.parse_args()

def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    chkpt_dir = os.path.abspath(args.checkpoint_dir)
    inp_dir = os.path.abspath(args.input_dir)
    out_dir = os.path.abspath(args.output_dir)
    ftype = args.type
    prefix = args.prefix
    tile_size = 256

    if not os.path.exists(chkpt_dir):
        raise RuntimeError("Checkpoint directory {} not found.".format(chkpt_dir))

    if not os.path.exists(inp_dir):
        raise RuntimeError("Input directory {} not found.".format(inp_dir))

    if not os.path.exists(out_dir):
        logging.info("Created output directory {}".format(out_dir))
        makedirs(out_dir)

    # Load images in input data directory one at a time.
    # For each image, do the following:
    #   Calculate its dimensions
    #   Apply padding to make it divisible into 256x256 tiles
    #   Save tiles in a separate directory
    #   Run GAN on all tiles
    #   Compile tiles back into one image and save it in the results directory

    logging.info("Collecting list of input files.")
    input_files = glob('{}/{}*.{}'.format(inp_dir, prefix, ftype))

    for file in input_files:
        logging.info("Parsing image file {}".format(file))
        img = cv.imread(file).astype(np.float32)
        logging.info("Image has shape: {}".format(img.shape))
        fname = os.path.basename(file)

        width, height, _ = img.shape

        pwidth = ceil(width / tile_size) * tile_size
        pheight = ceil(height / tile_size) * tile_size

        logging.info("Converting image of size {}x{} to size {}x{}".format(width, height, pwidth, pheight))

        pad_dir = "/".join([inp_dir, "padded"])

        if not os.path.exists(pad_dir):
            makedirs(pad_dir)
            logging.info("Created output directory {}".format(pad_dir))

        pfile = "/".join([pad_dir, fname])

        cmd = [
            "convert", file,
            "-gravity", "center",
            "-backgroundout_dir", "black",
            "-extent", "{}x{}".format(pwidth, pheight),
            "+repage",
            pfile
        ]

        subprocess.run(cmd)
        logging.info("Finished padding file.")

        fname_pure = fname.split(".")[0]
        # tile_dir = '/'.join([inp_dir, "tiles", fname_pure])
        tile_dir = './dataset/temp'
        gan_stupidity_dirs = [
            tile_dir,
            "/".join([tile_dir, "style"]),
            "/".join([tile_dir, "smooth"])
        ]

        for dir in gan_stupidity_dirs:
            if not os.path.exists(dir):
                makedirs(dir)
                logging.info("Created output directory {}".format(os.path.abspath(dir)))
            else:
                logging.info("Overwriting existing output directory {}".format(os.path.abspath(dir)))

        tfilename = ".".join(["tile-%d", ftype])
        tfilepath =  "/".join([tile_dir, tfilename])

        cmd = [
            "convert", pfile,
            "-crop", "256x256",
            "+repage",
            tfilepath
        ]

        # Confirmed: For some reason, the tiles are being generated from the top-right corner, to the bottom-right,
        # and so on until the bottom-left.

        subprocess.run(cmd)
        logging.info("Finished generating tiles at {}.".format(os.path.abspath(tile_dir)))

        # gan_output_dir = "/".join([out_dir, fname_pure, "tiles"])
        gan_output_dir = "temp"
        # if not os.path.exists(gan_output_dir):
        #     makedirs(gan_output_dir)
        #     logging.info("Created temporary output directory for GAN: {}".format(os.path.abspath(gan_output_dir)))
        # else:
        #     logging.info("Re-using existing temporary output directory for GAN: {}".format(os.path.abspath(
        #         gan_output_dir)))

        cmd = [
            "python", "test.py",
            "--checkpoint_dir", chkpt_dir,
            "--test_dir", tile_dir,
            "--output_dir", gan_output_dir,
        ]

        # subprocess.run(cmd, check=True)

        nrows = ceil(height / tile_size)
        ncols = ceil(width / tile_size)
        ntiles = nrows * ncols

        sel = []
        for col in range(ncols - 1, -1, -1):
            for row in range(nrows):
                sel.append(col + row * ncols)

        # The indices of sel correspond to the names assigned to tiles at these respective row-major form  indices

        ids = range(ntiles) # Generate list of the above indices
        t = list(zip(sel, ids))
        t.sort(key=lambda t: t[0])  # Sort the concatenated list according to values of sel instead
        _, ids = list(zip(*t)) # Now the list of indices has become a sorted list of names
        logging.info("Generated {} sorted ids from a matrix of {} rows and {} cols".format(len(ids), nrows, ncols))
        logging.info("Row 0 Col 0:tile-{}\nRow 0 Col n:tile{}\nRow n Col 0:tile-{}\nRow n Col n:tile{}\n".format(
            ids[0], ids[ncols-1], ids[(ncols - 1)* nrows], ids[nrows * ncols - 1]
        ))

        idx = lambda irow, icol: irow * ncols + icol


        filenames = ["results/{}/tile-{}.{}".format(gan_output_dir, i, ftype) for i in ids]
        # filenames = ["results/{}/tile-{}.{}".format(gan_output_dir, ids[idx(r, c)], ftype) for c in [0, 1,] for r in [0, 1,]]

        for f in filenames:
            if not os.path.isfile(f):
                logging.info("Could not find {}".format(f))

        logging.info("Verified all filenames, such as {}".format(filenames[0]))
        filenames = " ".join(filenames)
        composite_file = "/".join([out_dir, "{}.{}".format(fname_pure, ftype)])

        cmd = [
            "montage",
            # "results/temp/tile-*",
            "-mode", "concatenate",
            "-geometry", "{}x{}".format(tile_size, tile_size),
            "-tile", "{}x{}+0+0".format(ncols, nrows),
            filenames,
            composite_file
        ]

        # subprocess.run(cmd, check=True)
        with open("{}/filenames".format(out_dir), "w") as fn:
        #     # fn.write(" ".join(cmd))
            fn.write(filenames)

if __name__ == '__main__':
    main()