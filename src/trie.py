import torch


class RouteConstraint:
    def __init__(self, routes, eos, vocab_size, device):
        children = [{}]
        for route in routes:
            state = 0
            for token in route:
                if token not in children[state]:
                    children[state][token] = len(children)
                    children.append({})
                state = children[state][token]
            children[state][eos] = -1
        keys, targets, allowed, offsets = [], [], [], [0]
        for state, edges in enumerate(children):
            for token, target in sorted(edges.items()):
                keys.append(state * vocab_size + token)
                targets.append(target)
                allowed.append(token)
            offsets.append(len(allowed))
        self.keys = torch.tensor(keys, device=device)
        self.targets = torch.tensor(targets, device=device)
        self.allowed = torch.tensor(allowed, device=device)
        self.offsets = torch.tensor(offsets, device=device)
        self.vocab_size = vocab_size
        self.eos = eos

    def __call__(self, input_ids, scores):
        states = torch.zeros(len(input_ids), dtype=torch.long, device=input_ids.device)
        valid = torch.ones_like(states, dtype=torch.bool)
        for token in input_ids[:, 1:].T:
            keys = states.clamp_min(0) * self.vocab_size + token
            pos = torch.searchsorted(self.keys, keys).clamp_max(len(self.keys) - 1)
            nxt = self.targets[pos]
            valid = valid & (self.keys[pos] == keys) & (nxt >= 0)
            states = torch.where(valid, nxt, states)
        starts = self.offsets[states]
        lengths = self.offsets[states + 1] - starts
        lengths = torch.where(valid, lengths, 0)
        rows = torch.repeat_interleave(torch.arange(len(scores), device=scores.device), lengths)
        bases = torch.repeat_interleave(starts, lengths)
        within = torch.arange(len(rows), device=scores.device) - torch.repeat_interleave(
            lengths.cumsum(0) - lengths, lengths
        )
        tokens = self.allowed[bases + within]
        masked = torch.full_like(scores, -torch.inf)
        masked[rows, tokens] = scores[rows, tokens]
        fallback = (~valid) | (lengths == 0)
        masked[fallback, self.eos] = scores[fallback, self.eos]
        return masked
