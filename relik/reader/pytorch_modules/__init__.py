# from relik.reader.pytorch_modules.hf.modeling_relik import RelikReaderSpanModel
# from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
# from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction


RELIK_READER_CLASS_MAP = {
    "RelikReaderSpanModel": "relik.reader.pytorch_modules.span.RelikReaderForSpanExtraction",
    "RelikReaderREModel": "relik.reader.pytorch_modules.triplet.RelikReaderForTripletExtraction",
}
