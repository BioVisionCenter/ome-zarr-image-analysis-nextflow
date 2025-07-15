#! /usr/bin/env python3

from pathlib import Path
from skimage import filters, transform, morphology
import zarr
from ome_zarr import writer, scale, reader
from ome_zarr.io import parse_url
import numpy as np
from typing import Optional
import fire
import os
from ngio import open_ome_zarr_container


def segment(
        omezarr_root: str,
        resolution: int = 0,
        channel: Optional[int] = 0,
        segmentation_name: str = 'otsu',
    ):
    ### Apply otsu threshold to OME-Zarr and write output to a separate OME-Zarr directory.
    # Read OME-Zarr and specify a resolution layer.
    ome_zarr = open_ome_zarr_container(omezarr_root)
    
    image = ome_zarr.get_image(path=str(resolution))
    
    # Get the a specific channel from the image.
    array = image.get_array(c=channel)
    
    # This won't be necessary in ngio>0.4 
    channel_axis = image.dimensions.get("c")
    array = np.squeeze(array, axis=channel_axis)
    
    # Apply Otsu's thresholding method to the image array.
    t = filters.threshold_otsu(array)
    mask = morphology.label(array > t).astype(np.uint8)
    
    # Create a new empty OME-Zarr label.
    label = ome_zarr.derive_label(name=segmentation_name, dtype="uint8")
    
    # Write the mask to the label.
    label.set_array(mask)
    
    # Propagate the changes to all resolutions.
    label.consolidate()
    

def version():
    print("0.0.1")

if __name__ == '__main__':
    cli = {
        "version": version,
        "run": segment
    }
    fire.Fire(cli)
