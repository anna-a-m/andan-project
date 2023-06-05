from collections import Counter

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

COLUMNS = ['Form', 'Lemma', 'Pos', 'Phrase', 'Hand', 'Phase', 'Practice', 'Semantic', 'HandShapeShape', 'PalmDirection',]

def get_gesture_features(phrase_info: pd.DataFrame) -> list:
    """Gesture feature generator
    Find gesture feature annotation that accompany gesture phrase 
    and return it as a dictionary
    Arguments:
        phrase_info: pd.DataFrame
    Yields:
        dict"""
    for row in phrase_info.iterrows():
        pref = '.'.join(row[1].layer.split('.')[:-1])
        hand = row[1].layer.split('.')[-2]
        gesture_query = whole.query(
            f"filename == '{row[1].filename}' & layer.str.contains('{pref}')"\
            +f"& ({row[1].start} <= start < {row[1].end})"
        )
        annotation_pairs = [(r[1].layer.replace(f"{pref}.", ''), r[1].annotation) for r in gesture_query.iterrows()]
        layer_c = Counter([x[0] for x in annotation_pairs])
        if len(layer_c) == len(annotation_pairs):
            d = {k: v for k, v in annotation_pairs}
            d['Hand'] = hand
            yield d
        else:
            duplicated_key = layer_c.most_common(1)[0][0]
            for dkv in [v for k, v in annotation_pairs if k == duplicated_key]:
                d = {k: v for k, v in annotation_pairs if k != duplicated_key}
                d['Hand'] = hand
                d[duplicated_key] = dkv
                yield d
                
def update_storage(_d_: dict):
    global COLUMNS

    to_write = [str(_d_[k]) if k in _d_ else str(np.nan) for k in COLUMNS]
    with open('SaGa++ Dataset//data.txt', 'a', encoding='utf-8') as file:
        file.write(','.join(to_write))
        file.write('\n')


whole = pd.read_csv('SaGa++ Dataset//all_info.txt', sep='\t', on_bad_lines='skip', names=['layer', 
                                                                                          'speaker',
                                                                                          'start', 
                                                                                          'end',
                                                                                          'duration',
                                                                                          'annotation',
                                                                                          'filename'])

whole.drop(axis=1, columns=['speaker'], inplace=True)
whole['speaker'] = ['Router' if b else 'Follower' for b in whole.layer.str.startswith('R.')]

whole.layer = whole.layer.apply(lambda x: x.strip())
whole.layer = whole.layer.apply(lambda x: 'R.G.Left Semantic' if x == 'R.G.Left Semactic' else x)
whole.layer = whole.layer.apply(lambda x: x.replace(' ', '.') if ' ' in x and ('Left' in x or 'Right' in x) else x)
whole.layer = whole.layer.apply(lambda x: x.replace(' ', '') if ' ' in x else x)

for s in tqdm(whole.query("layer.str.contains('R.S.Form') & annotation.notnull()").iterrows()):
    q = whole.query(f"`start` >= {s[1].start} & `start` < {s[1].end} & `filename` == '{s[1].filename}'"\
               + f"& `speaker` == '{s[1].speaker}' | `start` <= {s[1].start} & `end` >= {s[1].end}"\
               + f"& `filename` == '{s[1].filename}' & `speaker` == '{s[1].speaker}'")
    d = {
        'Form': q[q.layer.str.contains('Form')].annotation.values[0],
        'Lemma': q[q.layer.str.contains('Lemma')].annotation.values[0],
        'Pos': q[q.layer.str.contains('Pos')].sort_values(by='start').annotation.values[0],
    }
    
    q_phr = q[q.layer.str.contains('Phrase')]
    
    if q_phr.shape[0]:
        w = list(get_gesture_features(q_phr))
        _ = [d1.update(d) for d1 in w]
        _ = [update_storage(d1) for d1 in w]
    else:
        update_storage(d)
        

DEICTICA = '''anbei, anliegend, hier, oben, unten, rechts, links, hiermit, folgen, hier, da, 
dort, oben, unten, vor, hinter, hiesig, hin, her, kommen, gehen, holen, bringen, folgen, 
da, dort, hier, hierhin, dorthin, 
vorn, hinten, oben, unten, 
herauf, hinauf, 
darin, daraus, davor, dahinter, darüber, darunter, 
drin, drüber, drunter,'''

DEICTICA = list(set(DEICTICA.split(', ')))
DEICTICA = list(map(str.strip, DEICTICA))

def deictic_annotation(x):
    global DEICTICA
    if x in DEICTICA:
        return 'deictic'
    else:
        return 'non-deictic'
    
data = pd.read_csv('SaGa++ Dataset//data.txt', names=COLUMNS)
data['IsDeictic'] = data.Lemma.apply(deictic_annotation)
data.to_csv('SaGa++ Dataset//data.txt', index=False)
