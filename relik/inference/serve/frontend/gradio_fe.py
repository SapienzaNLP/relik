from functools import partial
import os

import gradio as gr
import requests
import spacy
from pyvis.network import Network
from spacy import displacy
from spacy.tokens import Doc, Span

from relik.common.utils import CONFIG_NAME, from_cache
from relik.inference.annotator import Relik
from relik.inference.data.objects import RelikOutput
from relik.inference.serve.frontend.utils import get_random_color


LOGO = """
<div style="text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/5f0b462819cb630495b814d7/NJPRSmlwi6sY1qQEIcgRV.png" style="max-width: 850px; height: auto;"> 
</div>
"""

DESCRIPTION = """
<div style="display:flex; justify-content: center; align-items: center; flex-direction: row;">
    <a href="https://2024.aclweb.org/"><img src="http://img.shields.io/badge/ACL-2024-4b44ce.svg"></a> &nbsp; &nbsp; 
    <a href="https://aclanthology.org/"><img src="http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg"></a> &nbsp; &nbsp; 
    <a href="https://arxiv.org/abs/2408.00103"><img src="https://img.shields.io/badge/arXiv-2408.00103-b31b1b.svg"></a>
</div>
<br>
<div style="display:flex; justify-content: center; align-items: center; flex-direction: row;">
    <a href="https://huggingface.co/collections/sapienzanlp/relik-retrieve-read-and-link-665d9e4a5c3ecba98c1bef19"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D"></a> &nbsp; &nbsp;
    <a href="https://github.com/SapienzaNLP/relik"><img src="https://img.shields.io/badge/GitHub-Repo-121013?logo=github&logoColor=white"></a> &nbsp; &nbsp;
    <a href="https://github.com/SapienzaNLP/relik/releases"><img src="https://img.shields.io/github/v/release/SapienzaNLP/relik"></a>
</div>
<br>
<div style="display:flex; justify-content: center; align-items: center; flex-direction: row;">
    <a href="https://nlp.uniroma1.it/"><img src="https://img.shields.io/badge/Sapienza NLP-802433.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAMAAABrrFhUAAADAFBMVEUAAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////8HPQsIAAAA/3RSTlMAAQIDBAUGBwgJCgsMDQ4PEBESExQVFhcYGRobHB0eHyAhIiMkJSYnKCkqKywtLi8wMTIzNDU2Nzg5Ojs8PT4/QEFCQ0RFRkdISUpLTE1O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"></a> &nbsp; &nbsp;
    <a href="https://babelscape.it/"><img src="https://img.shields.io/badge/Babelscape-215489.svg?logo=data:image/png;base64,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"></a>
</div>
<br>

<div style="text-align: center; display: flex; flex-direction: column; align-items: center;">

<h2>
    Retrieve, Read and LinK: Fast and Accurate Entity Linking and Relation Extraction on an Academic Budget
    <br>
    <span style="font-size: 0.9em;">
    A blazing fast and lightweight Information Extraction model for Entity Linking and Relation Extraction. 
    </span>
    <br>
    <span style="color: #919191; font-weight: 400; font-size: 0.8em;">
        <a href="https://riccardorlando.xyz/" style="color: #919191;" target="_blank">Riccardo Orlando</a>, 
        <a href="https://littlepea13.github.io/" style="color: #919191;" target="_blank">Pere-Lluís Huguet Cabot</a>, 
        <a href="https://edobobo.github.io/" style="color: #919191;" target="_blank">Edoardo Barba</a>, 
        and <a href="https://www.diag.uniroma1.it/navigli/" style="color: #919191;" target="_blank">Roberto Navigli</a> 
    </span>
<h2>
</div>
"""

INSTRUCTION = """
## Use it locally

Installation from PyPI

```bash
pip install relik
```

ReLiK is a lightweight and fast model for **Entity Linking** and **Relation Extraction**.
It is composed of two main components: a **retriever** and a **reader**.
The retriever is responsible for retrieving relevant documents from a large collection of documents,
while the reader is responsible for extracting entities and relations from the retrieved documents.
ReLiK can be used with the `from_pretrained` method to load a pre-trained pipeline.

Here is an example of how to use ReLiK for Entity Linking:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")

# RelikOutput(
#     text="Michael Jordan was one of the best players in the NBA.",
#     tokens=['Michael', 'Jordan', 'was', 'one', 'of', 'the', 'best', 'players', 'in', 'the', 'NBA', '.'],
#     id=0,
#     spans=[
#         Span(start=0, end=14, label="Michael Jordan", text="Michael Jordan"),
#         Span(start=50, end=53, label="National Basketball Association", text="NBA"),
#     ],
#     triplets=[],
#     candidates=Candidates(
#         span=[
#             [
#                 [
#                     {"text": "Michael Jordan", "id": 4484083},
#                     {"text": "National Basketball Association", "id": 5209815},
#                     {"text": "Walter Jordan", "id": 2340190},
#                     {"text": "Jordan", "id": 3486773},
#                     {"text": "50 Greatest Players in NBA History", "id": 1742909},
#                     ...
#                 ]
#             ]
#         ]
#     ),
# )
```

and for Relation Extraction:

```python
from relik import Relik
from relik.inference.data.objects import RelikOutput

relik = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-large")
relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")
```

For more information, please refer to the [source code](https://github.com/SapienzaNLP/relik/).
"""


# relik_available_models = [
#     "relik-ie/relik-reader-small-cie-wikipedia",
# ]

# relik_models = {
#     "relik-ie/relik-reader-small-cie-wikipedia": Relik.from_pretrained(
#         "relik-ie/relik-reader-small-cie-wikipedia",
#         index_precision="bf16",
#         device="cuda",
#         reader_kwargs={"dataset_kwargs": {"use_nme": True}},
#     ),
# }


def get_span_annotations(response, doc):
    dict_ents = {}
    el_link_wrapper = "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css'><a href='https://en.wikipedia.org/wiki/{}' style='color: #414141'><i class='fa-brands fa-wikipedia-w fa-xs' style='color: #414141'></i> {}</a>"

    spans = []
    for idx, span in enumerate(response.spans):
        spans.append(
            Span(
                doc,
                span.start,
                span.end,
                (
                    el_link_wrapper.format(span.label.replace(" ", "_"), span.label)
                    if span.label != "--NME--"
                    else "--NME--"
                ),
                # kb_id=span.label.replace(" ", "_")
            )
        )
        dict_ents[(span.start, span.end)] = (
            span.label + str(idx),
            doc[span.start : span.end].text,
            span.label,
            span.label.replace(" ", "_"),
        )
    colors = get_random_color(set([span.label_ for span in spans]))
    return spans, colors, dict_ents


def generate_graph(
    spans, response, colors, dict_ents, bgcolor="#111827", font_color="white"
):
    g = Network(
        width="720px",
        height="600px",
        directed=True,
        notebook=False,
        bgcolor=bgcolor,
        font_color=font_color,
    )
    g.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=50,
        spring_strength=0.001,
        damping=0.09,
        overlap=0,
    )

    for ent in spans:
        # if not NME use title:
        if dict_ents[(ent.start, ent.end)][2] != "--NME--":
            g.add_node(
                dict_ents[(ent.start, ent.end)][2],
                label=dict_ents[(ent.start, ent.end)][2],
                color=colors[ent.label_],
                title=dict_ents[(ent.start, ent.end)][2],
                size=15,
                labelHighlightBold=True,
            )
        else:
            g.add_node(
                ent.text,
                label=ent.text,
                color=colors[ent.label_],
                title=ent.text,
                size=15,
                labelHighlightBold=True,
            )
    seen_rels = set()
    for rel in response.triplets:
        if (
            dict_ents[(rel.subject.start, rel.subject.end)][2] == "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] == "--NME--"
        ):
            if (rel.subject.text, rel.object.text, rel.label) in seen_rels:
                continue
        elif (
            dict_ents[(rel.subject.start, rel.subject.end)][2] == "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] != "--NME--"
        ):
            if (
                rel.subject.text,
                dict_ents[(rel.object.start, rel.object.end)][2],
                rel.label,
            ) in seen_rels:
                continue
        elif (
            dict_ents[(rel.subject.start, rel.subject.end)][2] != "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] == "--NME--"
        ):
            if (
                dict_ents[(rel.subject.start, rel.subject.end)][2],
                rel.object.text,
                rel.label,
            ) in seen_rels:
                continue
        else:
            if (
                dict_ents[(rel.subject.start, rel.subject.end)][2],
                dict_ents[(rel.object.start, rel.object.end)][2],
                rel.label,
            ) in seen_rels:
                continue

        g.add_edge(
            (
                dict_ents[(rel.subject.start, rel.subject.end)][2]
                if dict_ents[(rel.subject.start, rel.subject.end)][2] != "--NME--"
                else dict_ents[(rel.subject.start, rel.subject.end)][1]
            ),
            (
                dict_ents[(rel.object.start, rel.object.end)][2]
                if dict_ents[(rel.object.start, rel.object.end)][2] != "--NME--"
                else dict_ents[(rel.object.start, rel.object.end)][1]
            ),
            label=rel.label,
            title=rel.label,
        )
        if (
            dict_ents[(rel.subject.start, rel.subject.end)][2] != "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] != "--NME--"
        ):
            seen_rels.add(
                (
                    dict_ents[(rel.subject.start, rel.subject.end)][2],
                    dict_ents[(rel.object.start, rel.object.end)][2],
                    rel.label,
                )
            )
        elif (
            dict_ents[(rel.subject.start, rel.subject.end)][2] != "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] == "--NME--"
        ):
            seen_rels.add(
                (
                    dict_ents[(rel.subject.start, rel.subject.end)][2],
                    rel.object.text,
                    rel.label,
                )
            )
        elif (
            dict_ents[(rel.subject.start, rel.subject.end)][2] == "--NME--"
            and dict_ents[(rel.object.start, rel.object.end)][2] != "--NME--"
        ):
            seen_rels.add(
                (
                    rel.subject.text,
                    dict_ents[(rel.object.start, rel.object.end)][2],
                    rel.label,
                )
            )
        else:
            seen_rels.add((rel.subject.text, rel.object.text, rel.label))
    # g.show(filename, notebook=False)
    html = g.generate_html()
    # need to remove ' from HTML
    html = html.replace("'", '"')

    return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""


RELIK = os.getenv("RELIK", "localhost:8000/api/relik")


def text_analysis(Text, Relation_Threshold, Window_Size, Window_Stride):
    relik = RELIK
    nlp = spacy.blank("xx")
    # annotated_text = relik(
    #     Text,
    #     annotation_type="word",
    #     num_workers=0,
    #     remove_nmes=False,
    #     relation_threshold=Relation_Threshold,
    #     window_size=Window_Size,
    #     window_stride=Window_Stride,
    # )
    print(f"Using ReLiK at {relik}")
    print(
        f"Querying ReLiK with ?text={Text}&relation_threshold={Relation_Threshold}&window_size={Window_Size}&window_stride={Window_Stride}&annotation_type=word&remove_nmes=False"
    )
    response = requests.get(
        f"http://{relik}/?text={Text}&relation_threshold={Relation_Threshold}&window_size={Window_Size}&window_stride={Window_Stride}&annotation_type=word&remove_nmes=False",
    )
    if response.status_code != 200:
        raise gr.Error(response.text)
    annotated_text = RelikOutput(**response.json())
    doc = Doc(nlp.vocab, words=[token.text for token in annotated_text.tokens])
    spans, colors, dict_ents = get_span_annotations(response=annotated_text, doc=doc)
    doc.spans["sc"] = spans

    # build the EL display
    display_el = displacy.render(
        doc, style="span", options={"colors": colors}
    )  # , "kb_url_template": "https://en.wikipedia.org/wiki/{}"})
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
    if annotated_text.triplets:
        # background_color should be the same as the background of the page
        display_re = generate_graph(spans, annotated_text, colors, dict_ents)
    return display_el, display_re


theme = theme = gr.themes.Base(
    primary_hue="rose",
    secondary_hue="rose",
    text_size="lg",
    # font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
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
    white-space: nowrap;
}
"""

with gr.Blocks(fill_height=True, css=css, theme=theme) as demo:
    # check if demo is running in dark mode
    gr.Markdown(LOGO)
    gr.Markdown(DESCRIPTION)
    gr.Interface(
        text_analysis,
        [
            gr.Textbox(label="Input Text", placeholder="Enter sentence here..."),
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.5,
                label="Relation Threshold",
                info="Minimum confidence for relation extraction (Only for RE and cIE)",
            ),
            gr.Slider(
                minimum=16,
                maximum=128,
                step=16,
                value=64,
                label="Window Size",
                info="Window size for the sliding window",
            ),
            gr.Slider(
                minimum=8,
                maximum=64,
                step=8,
                value=32,
                label="Window Stride",
                info="Window stride for the sliding window",
            ),
        ],
        [gr.HTML(label="Entities"), gr.HTML(label="Relations")],
        examples=[
            [
                "Avram Noam Chomsky  born December 7, 1928) is an American professor and public intellectual known for his work in linguistics, political activism, and social criticism. Sometimes called 'the father of modern linguistics', Chomsky is also a major figure in analytic philosophy and one of the founders of the field of cognitive science. He is a laureate professor of linguistics at the University of Arizona and an institute professor emeritus at the Massachusetts Institute of Technology (MIT). Among the most cited living authors, Chomsky has written more than 150 books on topics such as linguistics, war, and politics. In addition to his work in linguistics, since the 1960s Chomsky has been an influential voice on the American left as a consistent critic of U.S. foreign policy, contemporary capitalism, and corporate influence on political institutions and the media."
            ],
            [
                "'Bella ciao' (Italian pronunciation: [ˈbɛlla ˈtʃaːo]; 'Goodbye beautiful') is an Italian song dedicated to the partisans of the Italian resistance, which fought against the occupying troops of Nazi Germany and the collaborationist Fascist forces during the liberation of Italy. It was based on a folk song of the late 19th century, sung by female workers of the paddy fields in Northern Italy (mondine) in protest against harsh working conditions. Versions of 'Bella ciao' continue to be sung worldwide as a hymn of resistance."
            ],
        ],
        allow_flagging="never",
    )
    gr.Markdown("")
    gr.Markdown(INSTRUCTION)


def main(server_port: int = 8001):
    demo.launch(server_port)


if __name__ == "__main__":
    main()
