"""The 24-way OOF merger validates model-level provenance and TaxID coverage."""
import csv
import json
from pathlib import Path
import pytest
from vh3.common import FEATURE_DIMS,write_ids
from vh3.merge_parts import HEADER, merge,validate_merge,validate_one
from vh3.train import MODELS


def test_merge_all_24_combinations(tmp_path):
    length='1000bp'
    ids={'Insect':[str(100+i) for i in range(5)],
         'Plant':[str(200+i) for i in range(5)],
         'Other':[str(300+i) for i in range(5)]}
    featroot=tmp_path/'features'
    parts=tmp_path/'parts'
    for feat in FEATURE_DIMS:
        for cl,cl_ids in ids.items():
            d=featroot/length/feat/cl
            d.mkdir(parents=True)
            write_ids(d/'ids.tsv',[(s,cl) for s in cl_ids])
        for model in MODELS:
            dest=parts/length/f'{feat}_{model}'
            dest.mkdir(parents=True)
            (dest/'run_config.json').write_text(json.dumps({'length':length,'features':[feat],'models':[model]}))
            with (dest/'oof_predictions.tsv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=HEADER,delimiter='\t')
                writer.writeheader()
                for cl,cl_ids in ids.items():
                    for fold,tid in enumerate(cl_ids,1):
                        writer.writerow({'Feature':feat,'Model':model,'Fold':fold,'taxid':tid,
                                        'true_class':cl,'eval_true_class':cl,'pred_class':cl,
                                        'is_correct':1,'allowed_classes':cl,
                                        'prob_Insect':'1.0' if cl=='Insect' else '0.0',
                                        'prob_Plant':'1.0' if cl=='Plant' else '0.0',
                                        'prob_Other':'1.0' if cl=='Other' else '0.0'})
            with (dest/'fold_audit.tsv').open('w',newline='') as f:
                w=csv.writer(f,delimiter='\t');w.writerow(['Fold']);w.writerows([[v] for v in range(1,6)])
    out=tmp_path/'merged'
    merge(parts,length,out,featroot)
    assert validate_merge(out,length)==24*15
    with pytest.raises(FileExistsError):merge(parts,length,out,featroot)
    # A truncated or mislabeled individual component must fail validation.
    d=parts/length/'1mer_GaussianNB'
    c=json.loads((d/'run_config.json').read_text())
    c['models']=['SVM']
    (d/'run_config.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='configuration'):
        validate_one(d,length,'1mer','GaussianNB',set(sum(ids.values(),[])))
