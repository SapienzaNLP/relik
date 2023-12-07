import os
from datetime import datetime as dt
from pathlib import Path

import requests
import spacy
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from spacy import displacy
from spacy.tokens import Doc
from streamlit_extras.badges import badge
from streamlit_extras.stylable_container import stylable_container
from utils import get_random_color, visualize_parser

from relik import Relik

# RELIK = os.getenv("RELIK", "localhost:8000/api/relik")

state_variables = {"has_run_free": False, "html_free": ""}


def init_state_variables():
    for k, v in state_variables.items():
        if k not in st.session_state:
            st.session_state[k] = v


def free_reset_session():
    for k in state_variables:
        del st.session_state[k]


def generate_graph(dict_ents, response, filename, options):
    g = Network(
        width="720px",
        height="600px",
        directed=True,
        notebook=False,
        bgcolor="#222222",
        font_color="white",
    )
    g.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=50,
        spring_strength=0.001,
        damping=0.09,
        overlap=0,
    )
    for ent in dict_ents:
        g.add_node(
            dict_ents[ent][0],
            label=dict_ents[ent][1],
            color=options["colors"][dict_ents[ent][0]],
            title=dict_ents[ent][0],
            size=15,
            labelHighlightBold=True,
        )

    for rel in response.triples:
        g.add_edge(
            dict_ents[(rel.subject.start, rel.subject.end)][0],
            dict_ents[(rel.object.start, rel.object.end)][0],
            label=rel.label,
            title=rel.label,
        )
    g.show(filename, notebook=False)


def set_sidebar(css):
    white_link_wrapper = (
        "<link rel='stylesheet' "
        "href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'><a href='{}'>{}</a>"
    )
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


def get_span_annotations(response):
    el_link_wrapper = (
        "<link rel='stylesheet' "
        "href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'>"
        "<a href='https://en.wikipedia.org/wiki/{}' style='color: #414141'><i class='fa-brands"
        " fa-wikipedia-w fa-xs'></i> <span style='font-size: 1.0em; font-family: monospace'> "
        "{}</span></a>"
    )
    tokens = response.tokens
    labels = ["O"] * len(tokens)
    dict_ents = {}
    # make BIO labels
    for idx, span in enumerate(response.spans):
        labels[span.start] = (
            "B-" + span.label + str(idx)
            if span.label == "NME"
            else "B-" + el_link_wrapper.format(span.label.replace(" ", "_"), span.label)
        )
        for i in range(span.start + 1, span.end):
            labels[i] = (
                "I-" + span.label + str(idx)
                if span.label == "NME"
                else "I-"
                + el_link_wrapper.format(span.label.replace(" ", "_"), span.label)
            )
        dict_ents[(span.start, span.end)] = (
            span.label + str(idx),
            " ".join(tokens[span.start : span.end]),
        )
    unique_labels = set(w[2:] for w in labels if w != "O")
    options = {"ents": unique_labels, "colors": get_random_color(unique_labels)}
    return tokens, labels, options, dict_ents


@st.cache_resource()
def load_model():
    return Relik.from_pretrained("riccorl/relik-relation-extraction-nyt-small")


def set_intro(css):
    # intro
    st.markdown("# ReLik")
    st.markdown(
        "### Retrieve, Read and LinK: Fast and Accurate Entity Linking "
        "and Relation Extraction on an Academic Budget"
    )
    # st.markdown(
    #     "This is a front-end for the paper [Universal Semantic Annotator: the First Unified API "
    #     "for WSD, SRL and Semantic Parsing](https://www.researchgate.net/publication/360671045_Universal
    #     _Semantic_Annotator_the_First_Unified_API_for_WSD_SRL_and_Semantic_Parsing),
    #     which will be presented at LREC 2022 by "
    #     "[Riccardo Orlando](https://riccorl.github.io), [Simone Conia](https://c-simone.github.io/), "
    #     "[Stefano Faralli](https://corsidilaurea.uniroma1.it/it/users/stefanofaralliuniroma1it),
    #     and [Roberto Navigli](https://www.diag.uniroma1.it/navigli/)."
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
        value="Michael Jordan was one of the best players in the NBA.",
        height=200,
        max_chars=1500,
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

    if "relik_model" not in st.session_state.keys():
        st.session_state["relik_model"] = load_model()
    relik_model = st.session_state["relik_model"]
    init_state_variables()
    # ReLik API call

    # spacy for span visualization
    nlp = spacy.blank("xx")

    if submit:
        text = text.strip()
        if text:
            st.session_state["filename"] = str(dt.now().timestamp() * 1000) + ".html"

            with st.spinner(text="In progress"):
                response = relik_model(text, annotation_type="word", num_workers=0)
                # response = requests.post(RELIK, json=text)
                # if response.status_code != 200:
                #     st.error("Error: {}".format(response.status_code))
                # else:
                #     response = response.json()

                # EL
                st.markdown("####")
                st.markdown("#### Entities")
                tokens, labels, options, dict_ents = get_span_annotations(
                    response=response
                )
                doc = Doc(nlp.vocab, words=tokens, ents=labels)
                display_el = displacy.render(doc, style="ent", options=options)
                display_el = display_el.replace("\n", " ")
                # heuristic, prevents split of annotation decorations
                display_el = display_el.replace(
                    "border-radius: 0.35em;",
                    "border-radius: 0.35em; white-space: nowrap;",
                )
                with st.container():
                    st.write(display_el, unsafe_allow_html=True)

                # RE
                generate_graph(
                    dict_ents, response, st.session_state["filename"], options
                )
                HtmlFile = open(st.session_state["filename"], "r", encoding="utf-8")
                source_code = HtmlFile.read()
                st.session_state["html_free"] = source_code
                os.remove(st.session_state["filename"])
                st.session_state["has_run_free"] = True
        else:
            st.error("Please enter some text.")

    if st.session_state["has_run_free"]:
        st.markdown("#### Relations")
        components.html(st.session_state["html_free"], width=720, height=600)


if __name__ == "__main__":
    run_client()
