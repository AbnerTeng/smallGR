import re

TOKENS = [f"<r{level}_{code}>" for level in range(4) for code in range(256)]


def route_ids(tokenizer, route):
    tokens = route.split()
    if len(tokens) != 4 or any(
        re.fullmatch(fr"<r{level}_([0-9]+)>", token) is None
        or int(token.removesuffix(">").rsplit("_", 1)[1]) >= 256
        for level, token in enumerate(tokens)
    ):
        raise ValueError(f"Invalid RQ3 + suffix ID: {route!r}")
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.unk_token_id in ids:
        raise ValueError(f"RQ tokens missing from tokenizer: {route!r}")
    return ids


def validate_tokens(tokenizer):
    for token in TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids != [tokenizer.convert_tokens_to_ids(token)] or ids == [tokenizer.unk_token_id]:
            raise ValueError(f"RQ token is not atomic: {token}")
