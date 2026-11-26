"""Right-truncate formatted local SFT sequences; never shorten evaluation.

Prompt/marker labels remain masked. A truncated-away answer has zero labels;
ordinary trainer loss is retained, as in the public paper implementation.
"""
from collections.abc import Mapping

def build(*, tokenizer, formatting_prompts_func, response_template, dataset,
          max_seq_length=512, import_provider=None, require_sample_identity=False,
          federated=False):
    if type(max_seq_length) is not int or max_seq_length != 512:
        raise ValueError('This experiment requires local SFT max_seq_length=512')
    if not len(dataset):
        raise ValueError('Empty local SFT dataset')
    if getattr(tokenizer,'truncation_side','right')!='right':
        raise ValueError('Local SFT requires right truncation')
    from lora_classifier.probe_collection_runtime import _require_offline_environment
    _require_offline_environment()
    sequences=[];ends=[];original=[];instructions=[];responses=[];retained=[]
    for row in dataset:
        text=formatting_prompts_func({'instruction':[row['instruction']], 'response':[row['response']]})[0]
        start=text.rfind(row['response'])
        marker=text.rfind(response_template,0,start)
        if not row['response'] or start<0 or marker<0:
            raise ValueError('Cannot identify formatter response boundary')
        boundary=marker+len(response_template)
        def tokens(value,truncate=False):
            kw=dict(add_special_tokens=True,truncation=truncate,padding=False)
            if truncate:kw['max_length']=512
            return list(tokenizer(value,**kw)['input_ids'])
        full=tokens(text);prefix=tokens(text[:boundary]);ids=tokens(text,True)
        if ids!=full[:512]:
            raise ValueError('Tokenizer did not preserve the first 512 tokens')
        if full[:len(prefix)]!=prefix:
            raise ValueError('Formatter response token boundary is ambiguous')
        end=min(len(prefix),len(ids))
        sequences.append(ids);ends.append(end);original.append(len(full))
        instructions.append(len(tokenizer(row['instruction'],add_special_tokens=False,truncation=False)['input_ids']))
        responses.append(max(0,len(full)-len(prefix)));retained.append(len(ids)-end)
    from structural_completion import StructuralCompletionCollator
    collator=StructuralCompletionCollator(tokenizer,sequences,ends)
    raw=collator
    if federated:
        from federated_learning.formal_sft_trainer import StrictFederatedCompletionCollator
        collator=StrictFederatedCompletionCollator(raw,require_identity=require_sample_identity)
    data=dataset
    columns={'input_ids':sequences,'attention_mask':[[1]*len(x) for x in sequences],
             '_completion_marker_end':ends,'_original_sequence_tokens':original,
             '_retained_sequence_tokens':[len(x) for x in sequences],
             '_original_instruction_tokens':instructions,'_original_response_tokens':responses,
             '_retained_response_tokens':retained,
             '_response_truncated':[a<b for a,b in zip(retained,responses)]}
    for name,values in columns.items():data=data.add_column(name,values)
    marker_ids=list(tokenizer(response_template,add_special_tokens=False,truncation=False)['input_ids'])
    return dict(base_collator=collator,raw_completion_collator=raw,training_dataset=data,
                response_template_ids=marker_ids,validation_report=dict(
                    sample_count=len(data),marker_token_ids=marker_ids,marker_length=len(marker_ids),
                    minimum_trainable_response_tokens=min(retained),maximum_sequence_length=max(map(len,sequences)),
                    truncated_samples=sum(n>512 for n in original),zero_response_samples=sum(n==0 for n in retained),
                    all_samples_valid=True,all_samples_have_response_labels=all(n>0 for n in retained),
                    truncation_side='right',max_seq_length=512))
