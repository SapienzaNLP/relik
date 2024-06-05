from typing import List, Union

from relik.inference.data.splitters.base_sentence_splitter import BaseSentenceSplitter


class WindowSentenceSplitter(BaseSentenceSplitter):
    """
    A :obj:`WindowSentenceSplitter` that splits a text into windows of a given size.
    """

    def __init__(self, window_size: int, window_stride: int, *args, **kwargs) -> None:
        super(WindowSentenceSplitter, self).__init__()
        self.window_size = window_size
        self.window_stride = window_stride

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        **kwargs,
    ) -> Union[List[str], List[List[str]]]:
        """
        Tokenize the input into single words using SpaCy models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.

        Returns:
            :obj:`List[List[str]]`: The input doc split into sentences.
        """
        return self.split_sentences(texts)

    def split_sentences(self, text: str | List, *args, **kwargs) -> List[List]:
        """
        Splits a `text` into sentences.

        Args:
            text (:obj:`str`):
                Text to split.

        Returns:
            :obj:`List[str]`: The input text split into sentences.
        """

        if isinstance(text, str):
            text = text.split()
        sentences = []
        # if window_stride is zero, we don't need overlapping windows
        self.window_stride = (
            self.window_stride if self.window_stride != 0 else self.window_size
        )
        for i in range(0, len(text), self.window_stride):
            # if the last stride is smaller than the window size, then we can
            # include more tokens form the previous window.
            if i != 0 and i + self.window_size > len(text):
                overflowing_tokens = i + self.window_size - len(text)
                if overflowing_tokens >= self.window_stride:
                    break
                i -= overflowing_tokens
            involved_token_indices = list(
                range(i, min(i + self.window_size, len(text)))
            )
            window_tokens = [text[j] for j in involved_token_indices]
            sentences.append(window_tokens)
        return sentences
