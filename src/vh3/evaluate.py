"""Evaluate pooled OOF predictions, removing 14 cross-host TaxIDs for all combinations.

The historical final method rounds each probability to 6 decimals before macro-OvR AUC.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from .common import CROSS_HOST, CLASSES,CLASS_TO_ID,LENGTHS,PUBLISHED_EVAL_COUNTS

REQUIRED={'Feature','Model','Fold','taxid','true_class','pred_class','prob_Insect','prob_Plant','prob_Other'}

def evaluate_table(path,length,expect_published=False):
    df=pd.read_csv(path,sep='\t',dtype={'taxid':str},low_memory=False)
    if not REQUIRED.issubset(df.columns):raise ValueError(f'Missing OOF columns: {REQUIRED-set(df.columns)}')
    summaries=[];matrices=[];audit=[]
    for (feature,model),raw in df.groupby(['Feature','Model'],sort=False):
        if raw.taxid.isna().any() or raw.taxid.astype(str).str.strip().eq('').any():
            raise ValueError(f'{length}/{feature}/{model}: blank TaxID')
        if raw.taxid.duplicated().any():raise ValueError(f'{length}/{feature}/{model}: duplicated TaxID')
        cross=raw.taxid.isin(CROSS_HOST)
        ordinary=raw.loc[~cross].copy()
        if set(ordinary.Fold.astype(int))!={1,2,3,4,5}:
            raise ValueError(f'{length}/{feature}/{model}: incomplete folds')
        if not set(ordinary.true_class).issubset(CLASSES) or not set(ordinary.pred_class).issubset(CLASSES):
            raise ValueError('Unexpected class labels')
        if expect_published:
            counts=ordinary.true_class.value_counts()
            for cl,exp in PUBLISHED_EVAL_COUNTS[length].items():
                if int(counts.get(cl,0))!=exp:raise ValueError(f'{length}/{feature}/{model} {cl}: {counts.get(cl,0)} != {exp}')
        probs=ordinary[['prob_Insect','prob_Plant','prob_Other']].to_numpy(dtype=float)
        if not np.isfinite(probs).all() or (probs<0).any() or (probs>1+1e-8).any():raise ValueError('Invalid probabilities')
        if not np.allclose(probs.sum(axis=1),1,atol=2e-5,rtol=0):raise ValueError('Probability row sum !=1')
        probs=np.round(probs,6)
        yt=ordinary.true_class.map(CLASS_TO_ID).to_numpy(dtype=int)
        yp=ordinary.pred_class.map(CLASS_TO_ID).to_numpy(dtype=int)
        aucs=[float(roc_auc_score((yt==i).astype(int),probs[:,i])) for i in range(3)]
        row={'length':length,'Feature':feature,'Model':model,
             'AUC':float(np.mean(aucs)),'ACC':float(accuracy_score(yt,yp)),
             'Recall':float(recall_score(yt,yp,average='macro',zero_division=0)),
             'Precision':float(precision_score(yt,yp,average='macro',zero_division=0)),
             'F1':float(f1_score(yt,yp,average='macro',zero_division=0))}
        summaries.append(row)
        cm=confusion_matrix(yt,yp,labels=[0,1,2])
        for i,c in enumerate(CLASSES):
            matrices.append({'length':length,'Feature':feature,'Model':model,'TrueClass':c,
                             **{f'Pred_{CLASSES[j]}':int(cm[i,j]) for j in range(3)}})
        audit.append({'length':length,'Feature':feature,'Model':model,'raw_n':len(raw),'crosshost_removed':int(cross.sum()),
                      'evaluated_n':len(ordinary),'AUC_Insect':aucs[0],'AUC_Plant':aucs[1],'AUC_Other':aucs[2],
                      'AUC_method':'pooled five-fold OOF macro OvR; rounded probabilities to 6 decimals'})
    return pd.DataFrame(summaries),pd.DataFrame(matrices),pd.DataFrame(audit)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',required=True,help='Directory with 1000bp/,3000bp/,5000bp/,full_length/ OOF files')
    p.add_argument('--output',required=True)
    p.add_argument('--expect-published',action='store_true')
    a=p.parse_args(argv)
    root=Path(a.results_dir); summaries=[];cms=[];audits=[]
    for length in LENGTHS:
        path=root/length/'oof_predictions.tsv'
        if not path.exists():raise FileNotFoundError(path)
        s,c,au=evaluate_table(path,length,a.expect_published)
        summaries.append(s);cms.append(c);audits.append(au)
        print(f'[OK] {length}: {len(s)} model/feature combinations',flush=True)
    out=Path(a.output)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    for name,parts in [('feature_model_summary.tsv',summaries),('oof_confusion_matrix.tsv',cms),('audit_provenance.tsv',audits)]:
        pd.concat(parts,ignore_index=True).to_csv(out/name,sep='\t',index=False,float_format='%.6f')
    print(f'[COMPLETE] summary -> {out}')
if __name__=='__main__':main()
