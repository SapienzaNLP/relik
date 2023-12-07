from typing import List

NME_SYMBOL = "--NME--"


def get_special_symbols(num_entities: int) -> List[str]:
    return [NME_SYMBOL] + [f"[E-{i}]" for i in range(num_entities)]


def get_special_symbols_re(num_entities: int, use_nme: bool = False) -> List[str]:
    if use_nme:
        return [NME_SYMBOL] + [f"[R-{i}]" for i in range(num_entities)]
    else:
        return [f"[R-{i}]" for i in range(num_entities)]
