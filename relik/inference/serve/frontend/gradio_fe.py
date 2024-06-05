import os

import gradio as gr
import spacy
from pyvis.network import Network
from spacy import displacy
from spacy.tokens import Doc

from relik.inference.annotator import Relik
from relik.inference.serve.frontend.utils import get_random_color


relik_available_models = ["riccorl/relik-relation-extraction-nyt-small"]
loaded_model = None


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


def generate_graph(dict_ents, response, options):
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
    # g.show(filename, notebook=False)
    html = g.generate_html()
    # need to remove ' from HTML
    html = html.replace("'", '"')

    return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""


def text_analysis(Text, Model):
    global loaded_model
    if loaded_model is None or loaded_model["key"] != Model:
        relik = Relik.from_pretrained(Model)
        loaded_model = {"key": Model, "model": relik}
    else:
        relik = loaded_model["model"]
    # spacy for span visualization
    nlp = spacy.blank("xx")
    annotated_text = relik(Text, annotation_type="word", num_workers=0)
    # doc = nlp(text)
    # html = displacy.render(doc, style="dep", page=True)
    tokens, labels, options, dict_ents = get_span_annotations(response=annotated_text)
    # build the EL display
    doc = Doc(nlp.vocab, words=tokens, ents=labels)
    display_el = displacy.render(doc, style="ent", options=options)
    display_el = display_el.replace("\n", " ")
    # heuristic, prevents split of annotation decorations
    display_el = display_el.replace(
        "border-radius: 0.35em;",
        "border-radius: 0.35em; white-space: nowrap;",
    )
    display_el = display_el.replace(
        "span style",
        "span id='el' style",
    )
    display_re = ""
    if annotated_text.triples:
        display_re = generate_graph(dict_ents, annotated_text, options)
    return display_el, display_re


LOGO = """
<div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://drive.google.com/uc?export=view&id=1UwPIfBrG021siM9SBAku2JNqG4R6avs6" style="max-width: 850px; height: auto;"> 
</div>
"""

DESCRIPTION = """
<div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
<h1>Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget</h1>
<h2>A blazing fast and lightweight Information Extraction model for Entity Linking and Relation Extraction.</h2>
<p>Riccardo Orlando, Pere-Lluís Huguet Cabot, Edoardo Barba, Roberto Navigli<p>
</div>
"""

theme = theme = gr.themes.Base(
    primary_hue="rose",
    secondary_hue="rose",
)

css = """
h1 {
  text-align: center;
  display: block;
}
mark {
    color: black;
}
#el {
    color: black;
}
"""

with gr.Blocks(fill_height=True, css=css, theme=theme) as demo:
    gr.Markdown(LOGO)
    gr.Markdown(DESCRIPTION)
    gr.Interface(
        text_analysis,
        [
            gr.Textbox(label="Input Text", placeholder="Enter sentence here..."),
            gr.Dropdown(relik_available_models),
        ],
        [gr.HTML(label="Entities"), gr.HTML(label="Relations")],
        examples=[
            ["Micheal Jordan was one of the best players in the NBA."],
        ],
        allow_flagging="never",
    )

if __name__ == "__main__":
    demo.launch()
