"""Completion labels bound to formatter-derived token boundaries, never text search."""
import torch

class StructuralCompletionCollator:
    def __init__(self, tokenizer, sequences, marker_ends):
        if getattr(tokenizer, 'padding_side', 'right') != 'right':
            raise ValueError('structural completion requires right padding')
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError('pad token ID required')
        if len(sequences) != len(marker_ends):
            raise ValueError('boundary count mismatch')
        self.boundaries = {}
        for sequence, end in zip(sequences, marker_ends):
            key = tuple(sequence)
            if not 0 < end <= len(key):
                raise ValueError('invalid structural completion boundary')
            if key in self.boundaries and self.boundaries[key] != end:
                raise ValueError('identical tokens have conflicting structural boundaries')
            self.boundaries[key] = end

    def __call__(self, features):
        if not features:
            raise ValueError('empty batch')
        width = max(len(row['input_ids']) for row in features)
        ids, masks, labels = [], [], []
        for row in features:
            sequence = list(row['input_ids'])
            key = tuple(sequence)
            if key not in self.boundaries:
                raise ValueError('tokens are not bound to a validated structural boundary')
            if list(row.get('attention_mask', [1]*len(sequence))) != [1]*len(sequence):
                raise ValueError('expected unpadded input with all-one attention mask')
            end = self.boundaries[key]
            padding = width-len(sequence)
            ids.append(sequence + [self.pad_id]*padding)
            masks.append([1]*len(sequence)+[0]*padding)
            labels.append([-100]*end + sequence[end:] + [-100]*padding)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in [('input_ids',ids),('attention_mask',masks),('labels',labels)]}
