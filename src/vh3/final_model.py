"""Fit a *new* deployable classifier from freshly computed feature arrays.

This fit uses all ordinary TaxIDs, excluding the 14 cross-host records. It is
separate from the five CV fold models used for manuscript performance metrics.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import joblib
import sklearn
from imblearn.under_sampling import RandomUnderSampler
from .common import CROSS_HOST, CLASS_TO_ID, FEATURE_DIMS, LENGTHS, sha256sum
from .train import get_model, load_feature, SEED, MODELS


def fit_deploy_model(features_dir, length, feature, model_name, output, seed=SEED, slice_seed=None, window_mode=None):
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite deployable model directory: {output}')
    X, meta=load_feature(features_dir,length,feature)
    y=meta['class'].map(CLASS_TO_ID).to_numpy(dtype=int)
    keep=~meta['taxid'].isin(CROSS_HOST).to_numpy()
    counts={cl:int((meta.loc[keep,'class']==cl).sum()) for cl in CLASS_TO_ID}
    if min(counts.values())<5:
        raise ValueError('Need at least five records from every class')
    X=X[keep]; y=y[keep]
    sampler=RandomUnderSampler(sampling_strategy='all',random_state=seed)
    Xbal,ybal=sampler.fit_resample(X,y)
    clf=get_model(model_name,seed)
    clf.fit(Xbal,ybal)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.part-',dir=output.parent))
    try:
        modelpath=stage/'model.joblib'
        joblib.dump(clf,modelpath,compress=3)
        manifest={'length':length,'feature':feature,'algorithm':model_name,
                  'label_encoding':CLASS_TO_ID,'training_mode':'final refit separate from fivefold CV',
                  'removed_crosshost_taxids':sorted(CROSS_HOST),
                  'training_class_counts_before_undersampling':counts,
                  'training_rows_after_undersampling':int(len(ybal)),
                  'random_seed':seed,
                  'window_mode':window_mode,
                  'slice_seed':slice_seed if window_mode=='regenerate' else None,
                  'sampling':'RandomUnderSampler(sampling_strategy=all, random_state=seed)',
                  'model_input_dimension':FEATURE_DIMS[feature],
                  'classes_':list(map(int,clf.classes_ if hasattr(clf,'classes_') else clf.named_steps['clf'].classes_)),
                  'scikit_learn_version':sklearn.__version__,
                  'model_sha256':sha256sum(modelpath),
                  'note':'No held-out accuracy is estimated from this final refit; use pooled OOF CV.'}
        (stage/'model_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        stage.rename(output)
    finally:
        if stage.exists():
            for fp in stage.iterdir():fp.unlink()
            stage.rmdir()
    print(f'[FINAL MODEL] {length} {feature}+{model_name}; n={len(ybal)} -> {output}',flush=True)
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',required=True)
    p.add_argument('--length',required=True,choices=LENGTHS)
    p.add_argument('--feature',default='LucaVirus',choices=list(FEATURE_DIMS))
    p.add_argument('--algorithm',default='SVM',choices=MODELS)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--slice-seed',default=None)
    p.add_argument('--window-mode',choices=('regenerate','historical'),default=None)
    a=p.parse_args(argv)
    fit_deploy_model(a.features_dir,a.length,a.feature,a.algorithm,a.output,a.seed,a.slice_seed,a.window_mode)
if __name__=='__main__':main()
