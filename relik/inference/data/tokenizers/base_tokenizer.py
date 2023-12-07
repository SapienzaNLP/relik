from typing import List, Union

from relik.inference.data.objects import Word


class BaseTokenizer:
    """
    A :obj:`Tokenizer` splits strings of text into single words, optionally adds
    pos tags and perform lemmatization.
    """

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        **kwargs
    ) -> List[List[Word]]:
        """
        Tokenize the input into single words.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> List[Word]:
        """
        Implements splitting words into tokens.

        Args:
            text (:obj:`str`):
                Text to tokenize.

        Returns:
            :obj:`List[Word]`: The input text tokenized in single words.

        """
        raise NotImplementedError

    def tokenize_batch(self, texts: List[str]) -> List[List[Word]]:
        """
        Implements batch splitting words into tokens.

        Args:
            texts (:obj:`List[str]`):
                Batch of text to tokenize.

        Returns:
            :obj:`List[List[Word]]`: The input batch tokenized in single words.

        """
        return [self.tokenize(text) for text in texts]

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
