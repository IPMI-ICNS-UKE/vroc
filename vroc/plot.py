import base64
from io import BytesIO
from typing import List

import bokeh.plotting as bpl
import bokeh.transform as btr
import colorcet
import datashader as ds
import datashader.bundling as bd
import datashader.transfer_functions as tf
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from matplotlib import cm
from PIL import Image
from skimage import filters, img_as_ubyte, io, measure, morphology

from vroc.helper import rescale_range


def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format="png")
    return out.getvalue()


def b64_image_files(images, colormap="magma"):
    # floats should be in [0, 1] for cmap
    images = rescale_range(
        images, input_range=(0.0, np.percentile(images, 99)), output_range=(0, 1)
    )
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        png = to_png(img_as_ubyte(cmap(im)))
        url = "data:image/png;base64," + base64.b64encode(png).decode("utf-8")
        urls.append(url)
    return urls


def plot_embedding(
    embedding: np.ndarray,
    tooltip_images: List[np.ndarray],
    patients: List["str"],
    colors,
):
    tooltip = """
        <div>
            <div>
                <img
                src="@image_files" height="240" alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">@patients</span>
            </div>
        </div>
              """

    data = pd.DataFrame(embedding, columns=("x", "y"))

    image_files = b64_image_files(tooltip_images)
    data["image_files"] = image_files
    data["patients"] = patients
    data["colors"] = colors

    source = ColumnDataSource(data)
    hover = HoverTool(tooltips=tooltip)
    fig = bpl.figure(width=1200, height=1200)
    fig.add_tools(hover)
    point_size = 100.0 / np.sqrt(data.shape[0])
    fig.circle("x", "y", source=source, size=point_size, color="colors")
    bpl.show(fig)
