"""Write a reference comparison report for Figure 5, without changing results.

A regenerated dataset does not necessarily match historic windows; discrepancies
are reported by default. Use --strict to make a mismatch fail the run.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import pandas as pd
from .common import LENGTHS

EXPECTED_AUC={'1000bp':0.9629,'3000bp':0.9833,'5000bp':0.9884,'full_length':0.9877}
EXPECTED_CM={
 '1000bp':[[822,131,73],[300,1770,41],[2831,1111,26372]],
 '3000bp':[[812,67,35],[150,1345,16],[1478,544,27472]],
 '5000bp':[[769,45,12],[107,1221,12],[1180,326,26807]],
 'full_length':[[943,49,34],[136,1962,13],[1642,462,28210]],
}
def compare(final_dir):
    root=Path(final_dir)
    s=pd.read_csv(root/'feature_model_summary.tsv',sep='\t')
    c=pd.read_csv(root/'oof_confusion_matrix.tsv',sep='\t')
    records=[]
    for length in LENGTHS:
        q=s[(s.length==length)&(s.Feature=='LucaVirus')&(s.Model=='SVM')]
        if len(q)!=1:
            raise ValueError(f'{length}: expected exactly one LucaVirus+SVM summary')
        auc=float(q.AUC.iloc[0]); reference=EXPECTED_AUC[length]
        t=c[(c.length==length)&(c.Feature=='LucaVirus')&(c.Model=='SVM')]
        actual=t.set_index('TrueClass').reindex(['Insect','Plant','Other'])[['Pred_Insect','Pred_Plant','Pred_Other']]
        if actual.isna().any().any():raise ValueError(f'{length}: missing confusion matrix row')
        cm=actual.to_numpy(dtype=int).tolist();expected_cm=EXPECTED_CM[length]
        status=('MATCH' if (f'{auc:.4f}'==f'{reference:.4f}' and cm==expected_cm) else 'DIFFERENT')
        records.append({'length':length,'feature':'LucaVirus','model':'SVM',
                        'reference_auc':f'{reference:.4f}','current_auc':f'{auc:.4f}',
                        'auc_match':int(f'{auc:.4f}'==f'{reference:.4f}'),
                        'cm_match':int(cm==expected_cm),
                        'reference_cm':str(expected_cm),'current_cm':str(cm),'status':status})
    return records

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-dir',required=True)
    p.add_argument('--report',default=None,help='Save TSV comparison; default inside final-dir')
    p.add_argument('--strict',action='store_true',help='Exit nonzero if any values differ')
    a=p.parse_args(argv)
    result=compare(a.final_dir)
    path=Path(a.report or (Path(a.final_dir)/'Figure5_reference_comparison.tsv'))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Refusing to overwrite report: {path}')
    with path.open('w',encoding='utf-8',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(result[0]),delimiter='\t');w.writeheader();w.writerows(result)
    errors=[r['length'] for r in result if r['status']!='MATCH']
    print(f'[REFERENCE] {len(result)-len(errors)}/4 match; report={path}; differing={errors}')
    if a.strict and errors:
        raise SystemExit(f'Figure5 strict comparison failed for: {errors}')
if __name__=='__main__':main()
