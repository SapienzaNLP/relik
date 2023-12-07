import base64
import random
from typing import Dict, List, Optional, Union

import spacy
import streamlit as st
from spacy import displacy


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def get_svg(svg: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def visualize_parser(
    doc: Union[spacy.tokens.Doc, List[Dict[str, str]]],
    *,
    title: Optional[str] = None,
    key: Optional[str] = None,
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
) -> None:
    """Visualizer for dependency parses.

    doc (Doc, List): The document to visualize.
    key (str): Key used for the streamlit component for selecting labels.
    title (str): The title displayed at the top of the parser visualization.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing parse information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See: https://spacy.io/api/top-level#options-dep
    """
    if displacy_options is None:
        displacy_options = dict()
    if title:
        st.header(title)
    docs = [doc]
    # add selected options to options provided by user
    # `options` from `displacy_options` are overwritten by user provided
    # options from the checkboxes
    for sent in docs:
        html = displacy.render(
            sent, options=displacy_options, style="dep", manual=manual
        )
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        st.write(get_svg(html), unsafe_allow_html=True)


def get_random_color(ents):
    colors = {}
    random_colors = generate_pastel_colors(len(ents))
    for ent in ents:
        colors[ent] = random_colors.pop(random.randint(0, len(random_colors) - 1))
    return colors


def floatrange(start, stop, steps):
    if int(steps) == 1:
        return [stop]
    return [
        start + float(i) * (stop - start) / (float(steps) - 1) for i in range(steps)
    ]


def hsl_to_rgb(h, s, l):
    def hue_2_rgb(v1, v2, v_h):
        while v_h < 0.0:
            v_h += 1.0
        while v_h > 1.0:
            v_h -= 1.0
        if 6 * v_h < 1.0:
            return v1 + (v2 - v1) * 6.0 * v_h
        if 2 * v_h < 1.0:
            return v2
        if 3 * v_h < 2.0:
            return v1 + (v2 - v1) * ((2.0 / 3.0) - v_h) * 6.0
        return v1

    # if not (0 <= s <= 1): raise ValueError, "s (saturation) parameter must be between 0 and 1."
    # if not (0 <= l <= 1): raise ValueError, "l (lightness) parameter must be between 0 and 1."

    r, b, g = (l * 255,) * 3
    if s != 0.0:
        if l < 0.5:
            var_2 = l * (1.0 + s)
        else:
            var_2 = (l + s) - (s * l)
        var_1 = 2.0 * l - var_2
        r = 255 * hue_2_rgb(var_1, var_2, h + (1.0 / 3.0))
        g = 255 * hue_2_rgb(var_1, var_2, h)
        b = 255 * hue_2_rgb(var_1, var_2, h - (1.0 / 3.0))

    return int(round(r)), int(round(g)), int(round(b))


def generate_pastel_colors(n):
    """Return different pastel colours.

    Input:
        n (integer) : The number of colors to return

    Output:
        A list of colors in HTML notation (eg.['#cce0ff', '#ffcccc', '#ccffe0', '#f5ccff', '#f5ffcc'])

    Example:
        >>> print generate_pastel_colors(5)
        ['#cce0ff', '#f5ccff', '#ffcccc', '#f5ffcc', '#ccffe0']
    """
    if n == 0:
        return []

    # To generate colors, we use the HSL colorspace (see http://en.wikipedia.org/wiki/HSL_color_space)
    start_hue = 0.0  # 0=red    1/3=0.333=green   2/3=0.666=blue
    saturation = 1.0
    lightness = 0.9
    # We take points around the chromatic circle (hue):
    # (Note: we generate n+1 colors, then drop the last one ([:-1]) because
    # it equals the first one (hue 0 = hue 1))
    return [
        "#%02x%02x%02x" % hsl_to_rgb(hue, saturation, lightness)
        for hue in floatrange(start_hue, start_hue + 1, n + 1)
    ][:-1]
