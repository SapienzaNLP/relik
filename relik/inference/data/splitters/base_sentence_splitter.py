from typing import List, Union


class BaseSentenceSplitter:
    """
    A `BaseSentenceSplitter` splits strings into sentences.
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
        raise NotImplementedError

    def split_sentences_batch(
        self, texts: List[str], *args, **kwargs
    ) -> List[List[str]]:
        """
        Default implementation is to just iterate over the texts and call `split_sentences`.
        """
        return [self.split_sentences(text) for text in texts]

    @staticmethod
    def check_is_batched(
        texts: Union[str, List[str], List[List[str]]], is_split_into_words: bool
    ):
        """
        Check if input is batched or a single sample.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to check.
            is_split_into_words (:obj:`bool`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`bool`: ``True`` if ``texts`` is batched, ``False`` otherwise.
        """
        return bool(
            (not is_split_into_words and isinstance(texts, (list, tuple)))
            or (
                is_split_into_words
                and isinstance(texts, (list, tuple))
                and texts
                and isinstance(texts[0], (list, tuple))
            )
        )
