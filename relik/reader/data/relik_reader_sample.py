import json
import numpy as np
from typing import Iterable


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class RelikReaderSample:
    def __init__(self, **kwargs):
        super().__setattr__("_d", {})
        self._d = kwargs

    def __getattribute__(self, item):
        return super(RelikReaderSample, self).__getattribute__(item)

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            # this is likely some python library-specific variable (such as __deepcopy__ for copy)
            # better follow standard behavior here
            raise AttributeError(item)
        elif item in self._d:
            return self._d[item]
        else:
            return None

    def __setattr__(self, key, value):
        if key in self._d:
            self._d[key] = value
        else:
            super().__setattr__(key, value)
    
    def to_jsons(self) -> str:
        if "predicted_window_labels" in self._d:
            new_obj = {
                k: v
                for k, v in self._d.items()
                if k != "predicted_window_labels" and k != "span_title_probabilities"
            }
            new_obj["predicted_window_labels"] = [
                [ss, se, pred_title]
                for (ss, se), pred_title in self.predicted_window_labels_chars
            ]
        else:
            return json.dumps(self._d, cls=NpEncoder)

    def to_dict(self) -> dict:
        return self._d


def load_relik_reader_samples(path: str) -> Iterable[RelikReaderSample]:
    with open(path) as f:
        for line in f:
            jsonl_line = json.loads(line.strip())
            relik_reader_sample = RelikReaderSample(**jsonl_line)
            yield relik_reader_sample
