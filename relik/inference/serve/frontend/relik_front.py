import os
from pathlib import Path

import requests
import streamlit as st
from spacy import displacy
from streamlit_extras.badges import badge
from streamlit_extras.stylable_container import stylable_container

RELIK = os.getenv("RELIK", "localhost:8000/api/entities")

import random


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
    start_hue = 0.6  # 0=red    1/3=0.333=green   2/3=0.666=blue
    saturation = 1.0
    lightness = 0.8
    # We take points around the chromatic circle (hue):
    # (Note: we generate n+1 colors, then drop the last one ([:-1]) because
    # it equals the first one (hue 0 = hue 1))
    return [
        "#%02x%02x%02x" % hsl_to_rgb(hue, saturation, lightness)
        for hue in floatrange(start_hue, start_hue + 1, n + 1)
    ][:-1]


def set_sidebar(css):
    white_link_wrapper = "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'><a href='{}'>{}</a>"
    with st.sidebar:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        st.image(
            "http://nlp.uniroma1.it/static/website/sapienza-nlp-logo-wh.svg",
            use_column_width=True,
        )
        st.markdown("## ReLiK")
        st.write(
            f"""
                - {white_link_wrapper.format("#", "<i class='fa-solid fa-file'></i>&nbsp; Paper")}
                - {white_link_wrapper.format("https://github.com/SapienzaNLP/relik", "<i class='fa-brands fa-github'></i>&nbsp; GitHub")}
                - {white_link_wrapper.format("https://hub.docker.com/repository/docker/sapienzanlp/relik", "<i class='fa-brands fa-docker'></i>&nbsp; Docker Hub")}
                """,
            unsafe_allow_html=True,
        )
        st.markdown("## Sapienza NLP")
        st.write(
            f"""
                - {white_link_wrapper.format("https://nlp.uniroma1.it", "<i class='fa-solid fa-globe'></i>&nbsp; Webpage")}
                - {white_link_wrapper.format("https://github.com/SapienzaNLP", "<i class='fa-brands fa-github'></i>&nbsp; GitHub")}
                - {white_link_wrapper.format("https://twitter.com/SapienzaNLP", "<i class='fa-brands fa-twitter'></i>&nbsp; Twitter")}
                - {white_link_wrapper.format("https://www.linkedin.com/company/79434450", "<i class='fa-brands fa-linkedin'></i>&nbsp; LinkedIn")}
                """,
            unsafe_allow_html=True,
        )


def get_el_annotations(response):
    # swap labels key with ents
    response["ents"] = response.pop("labels")
    label_in_text = set(l["label"] for l in response["ents"])
    options = {"ents": label_in_text, "colors": get_random_color(label_in_text)}
    return response, options


def set_intro(css):
    # intro
    st.markdown("# ReLik")
    st.markdown(
        "### Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget"
    )
    # st.markdown(
    #     "This is a front-end for the paper [Universal Semantic Annotator: the First Unified API "
    #     "for WSD, SRL and Semantic Parsing](https://www.researchgate.net/publication/360671045_Universal_Semantic_Annotator_the_First_Unified_API_for_WSD_SRL_and_Semantic_Parsing), which will be presented at LREC 2022 by "
    #     "[Riccardo Orlando](https://riccorl.github.io), [Simone Conia](https://c-simone.github.io/), "
    #     "[Stefano Faralli](https://corsidilaurea.uniroma1.it/it/users/stefanofaralliuniroma1it), and [Roberto Navigli](https://www.diag.uniroma1.it/navigli/)."
    # )
    badge(type="github", name="sapienzanlp/relik")
    badge(type="pypi", name="relik")


def run_client():
    with open(Path(__file__).parent / "style.css") as f:
        css = f.read()

    st.set_page_config(
        page_title="ReLik",
        page_icon="🦮",
        layout="wide",
    )
    set_sidebar(css)
    set_intro(css)

    # text input
    text = st.text_area(
        "Enter Text Below:",
        value="Obama went to Rome for a quick vacation.",
        height=200,
        max_chars=500,
    )

    with stylable_container(
        key="annotate_button",
        css_styles="""
            button {
                background-color: #802433;
                color: white;
                border-radius: 25px;
            }
            """,
    ):
        submit = st.button("Annotate")
    # submit = st.button("Run")

    # ReLik API call
    if submit:
        text = text.strip()
        if text:
            st.markdown("####")
            st.markdown("#### Entity Linking")
            with st.spinner(text="In progress"):
                response = requests.post(RELIK, json=text)
                if response.status_code != 200:
                    st.error("Error: {}".format(response.status_code))
                else:
                    response = response.json()

                    # Entity Linking
                    # with stylable_container(
                    #     key="container_with_border",
                    #     css_styles="""
                    #         {
                    #             border: 1px solid rgba(49, 51, 63, 0.2);
                    #             border-radius: 0.5rem;
                    #             padding: 0.5rem;
                    #             padding-bottom: 2rem;
                    #         }
                    #         """,
                    # ):
                    # st.markdown("##")
                    dict_of_ents, options = get_el_annotations(response=response)
                    display = displacy.render(
                        dict_of_ents, manual=True, style="ent", options=options
                    )
                    display = display.replace("\n", " ")
                    # wsd_display = re.sub(
                    #     r"(wiki::\d+\w)",
                    #     r"<a href='https://babelnet.org/synset?id=\g<1>&orig=\g<1>&lang={}'>\g<1></a>".format(
                    #         language.upper()
                    #     ),
                    #     wsd_display,
                    # )
                    with st.container():
                        st.write(display, unsafe_allow_html=True)

                    st.markdown("####")
                    st.markdown("#### Relation Extraction")

                    with st.container():
                        st.write("Coming :)", unsafe_allow_html=True)

        else:
            st.error("Please enter some text.")


if __name__ == "__main__":
    run_client()
