from typing import List, Union


class BlankSentenceSplitter:
    """
    A `BlankSentenceSplitter` splits strings into sentences.
    """

    def __call__(self, *args, **kwargs):
        """
        Calls :meth:`split_sentences`.
        """
        return self.split_sentences(*args, **kwargs)

    def split_sentences(
        self, text: str, max_len: int = 0, *args, **kwargs
    ) -> List[str]:
        """
        Splits a `text` :class:`str` paragraph into a list of :class:`str`, where each is a sentence.
        """
        return [text]

    def split_sentences_batch(
        self, texts: List[str], *args, **kwargs
    ) -> List[List[str]]:
        """
        Default implementation is to just iterate over the texts and call `split_sentences`.
        """
        return [self.split_sentences(text) for text in texts]
