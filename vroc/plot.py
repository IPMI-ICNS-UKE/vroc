from __future__ import annotations

import base64
import itertools
from io import BytesIO
from typing import List

import bokeh.plotting as bpl
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from skimage import img_as_ubyte

from vroc.common_types import IntTuple2D, Number, PathLike
from vroc.helper import rescale_range
from vroc.preprocessing import pad


class TiledArrayPlotter:
    def __init__(
        self,
        tiles: IntTuple2D,
        tile_shape: IntTuple2D,
        tile_padding: IntTuple2D = (10, 10),
    ):
        self.tiles = tiles
        self.tile_shape = tile_shape
        self.tile_padding = tile_padding
        self.padded_tile_shape = (
            self.tile_shape[0] + self.tile_padding[0],
            self.tile_shape[1] + self.tile_padding[1],
        )

        self.plot = np.zeros(
            (
                self.tiles[0] * self.padded_tile_shape[0],
                self.tiles[1] * self.padded_tile_shape[1],
                4,
            ),
            dtype=np.uint8,
        )
        # fill with black
        self.plot[:] = [0, 0, 0, 255]

        self.text = {}

    @staticmethod
    def _apply_cmap(
        array: np.ndarray,
        cmap: str,
        vmin: Number | None = None,
        vmax: Number | None = None,
    ) -> np.ndarray:
        cmap = cm.get_cmap(cmap)
        normalizer = colors.Normalize(vmin=vmin, vmax=vmax)
        return (cmap(normalizer(array)) * 255.0).astype(np.uint8)

    def _check_valid_tile(self, x_pos: int, y_pos: int):
        if x_pos >= self.tiles[0] or y_pos >= self.tiles[1]:
            raise ValueError(
                f"Tile position ({x_pos}, {y_pos}) is out of range "
                f"for plot with {self.tiles} tiles"
            )

    def add_array(
        self,
        row: int,
        col: int,
        array: np.ndarray,
        cmap: str,
        vmin: Number | None = None,
        vmax: Number | None = None,
    ):
        self._check_valid_tile(row, col)

        slicing = (
            slice(
                row * self.padded_tile_shape[0],
                (row + 1) * self.padded_tile_shape[0],
            ),
            slice(
                col * self.padded_tile_shape[1],
                (col + 1) * self.padded_tile_shape[1],
            ),
        )

        if vmin is None:
            vmin = array.min()
        if vmax is None:
            vmax = array.max()

        if array.shape != self.tile_shape:
            # pad array to fit into tile
            array = pad(array, target_shape=self.padded_tile_shape, pad_value=vmin)

        array = self._apply_cmap(array, cmap=cmap, vmin=vmin, vmax=vmax)

        self.plot[slicing] = array

    def add_text(
        self,
        row: int,
        col: int,
        text: str,
        font: str = "FreeMono.ttf",
        font_size: Number = 20,
    ):
        self._check_valid_tile(row, col)
        self.text[row, col] = {"text": text, "font": font, "font_size": font_size}

    def get_pil_image(self):
        image = Image.fromarray(self.plot, mode="RGBA")

        draw = ImageDraw.Draw(image)

        for x_pos, y_pos in self.text.keys():
            x_pixel_pos = x_pos * self.padded_tile_shape[0] + self.tile_padding[0]
            y_pixel_pos = y_pos * self.padded_tile_shape[1] + self.tile_padding[1]

            font_size = self.text[x_pos, y_pos]["font_size"]
            if font_size < 1.0:
                font_size = int(round(font_size * self.tile_shape[0]))

            font = ImageFont.truetype(self.text[x_pos, y_pos]["font"], size=font_size)
            draw.text(
                (y_pixel_pos, x_pixel_pos),
                text=self.text[x_pos, y_pos]["text"],
                font=font,
                fill=(255, 0, 0),
            )
        return image

    def save(self, filepath: PathLike):
        image = self.get_pil_image()
        with open(filepath, "wb") as f:
            image.save(f)


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
    tooltip_images: np.ndarray | List[np.ndarray],
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


if __name__ == "__main__":
    plot = TiledArrayPlotter((3, 4), tile_shape=(512, 512), tile_padding=(0, 10))
    for row in range(3):
        for col in range(4):
            plot.add_text(row, col, f"{row}, {col}", font_size=0.1)

    image = plot.get_pil_image()
    image.show()
