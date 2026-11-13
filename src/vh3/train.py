"""Train 3-class models from newly generated classwise feature arrays.

Reconstructs known historical 3mer/ESM2/LucaVirus training conventions.
Exact numeric paper reproduction still depends on historical slicing and original
pretrained checkpoint provenance; this runner NEVER accepts old feature imports.
"""
import argparse
import json
import sys
import csv
import tempfile
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import imblearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from .common import CLASSES,CLASS_TO_ID,FEATURE_DIMS,LENGTHS,CROSS_HOST,load_ids

SEED=20260914
MODELS=('RandomForest','KNN','SVM','GaussianNB')

def load_feature(root,length,feature):
    matrices=[]; metadata=[]
    for cl in CLASSES:
        d=Path(root)/length/feature/cl
        fi=d/'features.npy'; idsfile=d/'ids.tsv'
        if not fi.is_file() or not idsfile.is_file():
            raise FileNotFoundError(f'Missing feature files {fi} and/or {idsfile}')
        ids=load_ids(idsfile)
        if any(c!=cl for _,c in ids):raise ValueError(f'{idsfile}: expected class {cl}')
        arr=np.load(fi)
        if arr.shape!=(len(ids),FEATURE_DIMS[feature]):
            raise ValueError(f'{fi}: got shape {arr.shape}, expected {(len(ids),FEATURE_DIMS[feature])}')
        if not np.isfinite(arr).all():raise ValueError(f'{fi} has NaN/Inf')
        matrices.append(np.asarray(arr,dtype=np.float32));metadata+=ids
    tids=[tid for tid,_ in metadata]
    if len(set(tids))!=len(tids):raise ValueError('Duplicate TaxIDs across classes; stop instead of silently dropping')
    return np.vstack(matrices),pd.DataFrame(metadata,columns=['taxid','class'])

def build_folds(y,ids,seed=SEED,holdout_cross_host=True):
    """For legacy 3mer/ESM2/LucaVirus, cross-host records are only validation.
    For 1/2/4-mer, original 71.py used complete removal before model training.
    """
    ids=np.asarray(ids,dtype=str)
    cross=np.isin(ids,list(CROSS_HOST))
    if np.any(y[cross]!=CLASS_TO_ID['Insect']):
        raise ValueError('Cross-host TaxIDs should belong to the Insect source class')
    ordinary=np.flatnonzero(~cross)
    reserved=np.flatnonzero(cross)
    if np.min(np.bincount(y[ordinary],minlength=3))<5:
        raise ValueError('Need >=5 ordinary examples per class for stratified five-fold CV')
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    splits=list(splitter.split(np.zeros(len(ordinary)),y[ordinary]))
    held=np.array_split(np.random.RandomState(seed).permutation(reserved),5) if holdout_cross_host else [np.array([],dtype=int) for _ in range(5)]
    folds=[]
    for (tr,te),reserved_fold in zip(splits,held):
        folds.append((ordinary[tr],np.concatenate([ordinary[te],reserved_fold])))
    assert not any(np.isin(tr,reserved).any() for tr,_ in folds)
    return folds,cross

def get_model(name,seed=SEED):
    if name=='RandomForest':return RandomForestClassifier(n_estimators=500,random_state=seed,n_jobs=-1)
    if name=='KNN':return Pipeline([('scaler',StandardScaler()),('clf',KNeighborsClassifier(n_neighbors=5))])
    if name=='SVM':return Pipeline([('scaler',StandardScaler()),('clf',SVC(kernel='rbf',C=1.0,gamma='scale',probability=True,random_state=seed))])
    if name=='GaussianNB':return GaussianNB()
    raise ValueError(name)

def _dump_tsv(df,path):
    tmp=Path(str(path)+'.part')
    try:
        df.to_csv(tmp,sep='\t',index=False,float_format='%.6f')
        tmp.replace(path)
    finally:
        if tmp.exists():tmp.unlink()

def train(features_dir,length,output,feature_names,model_names,seed=SEED,expected=False,mode='legacy'):
    output=Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    if mode not in ('legacy','strict_exclude'):
        raise ValueError('mode must be legacy or strict_exclude')
    # Load and check ID alignment for all feature representations before training.
    # Historical 60.py used 3mer ID order as the authoritative five-fold order.
    ref_feature = '3mer' if '3mer' in feature_names else feature_names[0]
    _, ref = load_feature(features_dir,length,ref_feature)
    for f in feature_names:
        _,meta=load_feature(features_dir,length,f)
        if set(zip(meta.taxid,meta['class']))!=set(zip(ref.taxid,ref['class'])):
            raise ValueError(f'Feature {f} has TaxID/class set different from {ref_feature}')
    if expected:
        from .common import PUBLISHED_INPUT_COUNTS
        counts=ref['class'].value_counts()
        for cl, n in PUBLISHED_INPUT_COUNTS[length].items():
            if int(counts.get(cl,0))!=n:raise ValueError(f'{length}/{cl}: got {counts.get(cl,0)}, expected {n}')
    y=ref['class'].map(CLASS_TO_ID).to_numpy(dtype=int)
    folds,cross_mask=build_folds(y,ref.taxid.to_numpy(dtype=str),seed,holdout_cross_host=(mode=='legacy'))
    # For strict_exclude, delete cross-host rows from all model outputs via only ordinary validation folds.
    # For historical 1/2/4-mer, cross-host always absent from OOF, even in legacy mode.
    fold_rows=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f'.{output.name}.writing-',dir=output.parent))
    header=['Feature','Model','Fold','taxid','true_class','eval_true_class',
            'pred_class','is_correct','allowed_classes','prob_Insect','prob_Plant','prob_Other']
    stream=(staging/'oof_predictions.tsv').open('w',encoding='utf-8',newline='')
    writer=csv.DictWriter(stream,fieldnames=header,delimiter='\t')
    writer.writeheader()
    count_rows=0
    for f in feature_names:
        X,meta=load_feature(features_dir,length,f)
        order={t:i for i,t in enumerate(meta.taxid)}
        X=X[[order[t] for t in ref.taxid]]
        include_reserved=(mode=='legacy' and f in ('3mer','ESM2','LucaVirus'))
        use_folds=folds if include_reserved or mode=='strict_exclude' else build_folds(y,ref.taxid.to_numpy(dtype=str),seed,False)[0]
        for modelname in model_names:
            print(f'[TRAIN] {length} {f}+{modelname}',flush=True)
            for fold,(tr,va) in enumerate(use_folds,1):
                if not include_reserved:va=va[~cross_mask[va]]
                sampling=RandomUnderSampler(sampling_strategy='all',random_state=seed+fold)
                Xb,yb=sampling.fit_resample(X[tr],y[tr])
                model=get_model(modelname,seed)
                model.fit(Xb,yb)
                prob0=model.predict_proba(X[va])
                cls_order=model.classes_ if hasattr(model,'classes_') else model.named_steps['clf'].classes_
                prob=np.zeros((len(va),3),dtype=float)
                for j,c in enumerate(cls_order):prob[:,int(c)]=prob0[:,j]
                pred=np.argmax(prob,axis=1)
                fold_rows.append({'length':length,'Feature':f,'Model':modelname,'Fold':fold,
                                  'train_before':len(tr),'train_after':len(yb),'val_size':len(va),
                                  'held_out_cross_host_in_validation':int(cross_mask[va].sum())})
                for j,i in enumerate(va):
                    yt=int(y[i]);yp=int(pred[j]);is_cross=bool(cross_mask[i]);
                    accepted=is_cross and yp in (0,1)
                    eval_true=yp if accepted else yt
                    writer.writerow({'Feature':f,'Model':modelname,'Fold':fold,
                        'taxid':str(ref.iloc[i].taxid),'true_class':CLASSES[yt],
                        'eval_true_class':CLASSES[eval_true],'pred_class':CLASSES[yp],
                        'is_correct':int(eval_true==yp),'allowed_classes':'Insect|Plant' if is_cross else CLASSES[yt],
                        'prob_Insect':f'{prob[j,0]:.6f}','prob_Plant':f'{prob[j,1]:.6f}','prob_Other':f'{prob[j,2]:.6f}'})
                    count_rows+=1
    stream.close()
    _dump_tsv(pd.DataFrame(fold_rows),staging/'fold_audit.tsv')
    cfg={'length':length,'features':list(feature_names),'models':list(model_names),
         'classes':CLASS_TO_ID,'seed':seed,'cross_host_policy':mode,'fivefold':'StratifiedKFold(shuffle=True,random_state=seed)',
         'sampling':'RandomUnderSampler(sampling_strategy=all,random_state=seed+fold) applied inside training only',
         'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                     'scikit-learn':sklearn.__version__,'imbalanced-learn':imblearn.__version__}}
    (staging/'run_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    staging.rename(output)
    print(f'[COMPLETE] {length}: {count_rows} OOF entries -> {output}',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--length',choices=LENGTHS,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--features',default=','.join(FEATURE_DIMS),help='Comma list: 1mer,2mer,3mer,4mer,ESM2,LucaVirus')
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--mode',choices=['legacy','strict_exclude'],default='legacy')
    p.add_argument('--expect-published',action='store_true')
    a=p.parse_args(argv)
    f=a.features.split(',');m=a.models.split(',')
    if any(x not in FEATURE_DIMS for x in f) or any(x not in MODELS for x in m):p.error('Invalid features/models list')
    train(a.features_dir,a.length,a.output,f,m,a.seed,a.expect_published,a.mode)
if __name__=='__main__':main()
