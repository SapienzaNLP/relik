from relik import Relik
from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
from relik.retriever import GoldenRetriever
from relik.inference.data.objects import TaskType
from relik.reader.utils.relation_matching_eval import StrongMatchingPerRelation, StrongMatching
from relik.inference.data.objects import AnnotationType

import json
import pandas as pd
import argparse
import os
from relik.reader.data.relik_reader_sample import load_relik_reader_samples

def evaluate(reader_path, question_encoder_path, document_index_path, input_file, output_file, max_triplets=8, use_predefined_spans=False):
    # input_file path fileame bio-map.tsv
    mapping_classes = pd.read_csv(os.path.join(os.path.dirname(input_file), "bio-map.tsv"), sep="\t")
    mapping_dict = dict(zip(mapping_classes.label, mapping_classes.hierarchy))

    reader = RelikReaderForTripletExtraction(reader_path,
                                            dataset_kwargs={"use_nme": False, "max_triplets": max_triplets})
                        

    retriever = {
        "triplet": GoldenRetriever(
            question_encoder=question_encoder_path,
            document_index=document_index_path,
        ),
    }

    relik = Relik(reader=reader, retriever=retriever, top_k=max_triplets, task=TaskType.TRIPLET, device="cuda", window_size="sentence")

    samples = list(load_relik_reader_samples(input_file))
    # add text field to samples from joining words if there is no text field
    for id, sample in enumerate(samples):
        if sample._d.get("text") is None:
            sample._d["text"] = " ".join(sample.words)
        sample.doc_id = id
    results = relik(windows=samples, num_workers=4, device="cuda", progress_bar=True, annotation_type=AnnotationType.WORD, return_also_windows=True, use_predefined_spans=use_predefined_spans, relation_threshold=0.5)
    windows = []
    for sample in results:
        windows.extend(sample.windows)
    with open(output_file, "w") as f:
        for sample in windows:
            f.write(sample.to_jsons() + "\n")

    for sample in windows:
        for triplet in sample.predicted_relations:
            triplet["relation"]["name"] = mapping_dict[triplet["relation"]["name"]]

    strong_matching_metric = StrongMatchingPerRelation()
    results = list(windows)
    for k, v in strong_matching_metric(results).items():
        print(f"test_{k}", v)

    strong_matching_metric = StrongMatching()
    for k, v in strong_matching_metric(results).items():
        print(f"test_{k}", v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_path", type=str, required=True)
    parser.add_argument("--question_encoder_path", type=str, required=True)
    parser.add_argument("--document_index_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_triplets", type=int, default=8)
    parser.add_argument("--use_predefined_spans", action="store_true")
    args = parser.parse_args()
    evaluate(args.reader_path, args.question_encoder_path, args.document_index_path, args.input_file, args.output_file, args.max_triplets, args.use_predefined_spans)