"""AG News: reproducible linear and compact neural text classification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import ssl
import re
import time
import urllib.request

import certifi
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
BASE = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/'
SEED = 42


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)


def key(text):
    return re.sub(r'\s+', ' ', str(text).lower()).strip()


def clean(frame):
    frame = frame.copy()
    frame['text'] = (frame.title.fillna('') + ' ' + frame.description.fillna('')).str.strip()
    frame['text_key'] = frame.text.map(key)
    frame['title_key'] = frame.title.fillna('').map(key)
    counts = frame.groupby('text_key').label.nunique()
    conflicts = counts[counts > 1].index
    audit = {'input_rows': len(frame), 'empty_texts': int(frame.text_key.eq('').sum()),
             'duplicate_rows': int(frame.text_key.duplicated().sum()),
             'conflicting_texts': len(conflicts)}
    frame = frame[frame.label.isin([1, 2, 3, 4]) & frame.text_key.ne('') & ~frame.text_key.isin(conflicts)]
    frame = frame.drop_duplicates('text_key').copy()
    frame['group'] = frame.title_key.where(frame.title_key.ne(''), frame.text_key)
    audit['retained_rows'] = len(frame)
    return frame, audit


def prepare():
    DATA.mkdir(exist_ok=True)
    if (OUT / 'training.json').exists():
        raise SystemExit('Training already exists. Use the saved split; do not overwrite it during this experiment.')
    provenance = {}
    for name in ['train', 'test']:
        path = DATA / f'raw_{name}.csv'
        if not path.exists():
            print(f'Downloading {name} data...', flush=True)
            temporary = path.with_suffix('.download')
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(BASE + name + '.csv', context=context, timeout=60) as response:
                with temporary.open('wb') as destination:
                    shutil.copyfileobj(response, destination)
            temporary.replace(path)
        provenance[name] = {'url': BASE + name + '.csv',
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    raw = pd.read_csv(DATA / 'raw_train.csv', header=None, names=['label', 'title', 'description'])
    pool, audit = clean(raw)
    # Test TEXT is used only to audit overlap. Test labels never select training rows/settings.
    test = pd.read_csv(DATA / 'raw_test.csv', header=None, names=['label', 'title', 'description'])
    test_text = (test.title.fillna('') + ' ' + test.description.fillna('')).map(key)
    test_titles = test.title.fillna('').map(key)
    overlap = pool.text_key.isin(set(test_text)) | (pool.title_key.ne('') & pool.title_key.isin(set(test_titles)))
    audit['training_rows_removed_for_test_text_or_title_overlap'] = int(overlap.sum())
    pool = pool[~overlap].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(splitter.split(pool.text, pool.label, groups=pool.group))
    train, val = pool.iloc[tr], pool.iloc[va]
    assert set(train.group).isdisjoint(val.group)
    assert set(train.text_key).isdisjoint(val.text_key)
    for name, part in [('train', train), ('validation', val)]:
        part[['label', 'text', 'text_key', 'group']].to_csv(DATA / f'{name}.csv', index=False)
        audit[name] = {'rows': len(part), 'class_counts': part.label.value_counts().sort_index().to_dict()}
    save_json(OUT / 'data_audit.json', audit)
    save_json(OUT / 'data_sources.json', provenance)
    print(json.dumps(audit, indent=2))


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class TextData(Dataset):
    def __init__(self, texts, labels, vocab, length=120):
        self.rows = []
        for text, label in zip(texts, labels):
            ids = [vocab.get(t, 1) for t in tokens(text)[:length]] or [1]
            self.rows.append((ids, int(label) - 1))
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        return self.rows[index]


def collate(batch):
    sequences, labels = zip(*batch)
    x = torch.zeros(len(batch), max(map(len, sequences)), dtype=torch.long)
    for i, ids in enumerate(sequences):
        x[i, :len(ids)] = torch.tensor(ids)
    return x, torch.tensor(labels)


class NeuralClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4))
    def forward(self, x):
        mask = x.ne(0).unsqueeze(-1)
        pooled = (self.embedding(x) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)


def loader(frame, vocab, shuffle=False):
    return DataLoader(TextData(frame.text, frame.label, vocab), batch_size=256,
                      shuffle=shuffle, collate_fn=collate)


def neural_predictions(model, batches):
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in batches:
            values.extend((model(x).argmax(1) + 1).tolist())
    return np.array(values)


def report(name, split, frame, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    folder = OUT / split
    folder.mkdir(parents=True, exist_ok=True)
    results = classification_report(frame.label, predictions, labels=[1, 2, 3, 4],
                                    target_names=NAMES, output_dict=True, zero_division=0)
    save_json(folder / f'{name}_metrics.json', results)
    matrix = confusion_matrix(frame.label, predictions, labels=[1, 2, 3, 4])
    pd.DataFrame(matrix, index=NAMES, columns=NAMES).to_csv(folder / f'{name}_confusion.csv')
    ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(cmap='Blues', values_format='d')
    plt.title(f'{name} — {split}')
    plt.tight_layout()
    plt.savefig(folder / f'{name}_confusion.png', dpi=180)
    plt.close()
    errors = frame[['text', 'label']].copy()
    errors['predicted'] = predictions
    errors[errors.label != errors.predicted].head(100).to_csv(folder / f'{name}_errors.csv', index=False)
    return {'model': name, 'accuracy': results['accuracy'], 'macro_f1': results['macro avg']['f1-score']}


def train_models(epochs):
    if epochs < 1:
        raise SystemExit('--epochs must be at least 1')
    if (OUT / 'test').exists():
        raise SystemExit('Final test results already exist. Do not tune after observing test performance.')
    seed_all()
    train = pd.read_csv(DATA / 'train.csv', keep_default_na=False)
    val = pd.read_csv(DATA / 'validation.csv', keep_default_na=False)
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    dummy = DummyClassifier(strategy='most_frequent').fit(train.text, train.label)
    joblib.dump(dummy, OUT / 'dummy.joblib')
    rows = [report('dummy', 'validation', val, dummy.predict(val.text))]
    linear = Pipeline([
        ('tfidf', TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2),
                                max_features=50000, min_df=2, sublinear_tf=True)),
        ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
    print('Training TF-IDF + logistic regression...', flush=True)
    start = time.perf_counter()
    linear.fit(train.text, train.label)
    linear_seconds = time.perf_counter() - start
    joblib.dump(linear, OUT / 'linear.joblib')
    rows.append(report('linear', 'validation', val, linear.predict(val.text)))
    counts = Counter(t for text in train.text for t in tokens(text))
    vocab = {word: i + 2 for i, (word, _) in enumerate(counts.most_common(20000))}
    save_json(OUT / 'vocabulary.json', vocab)
    model = NeuralClassifier(len(vocab) + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    training_loader, validation_loader = loader(train, vocab, True), loader(val, vocab)
    best, stale, history = -1.0, 0, []
    print('Training compact neural classifier on CPU...', flush=True)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in training_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        pred = neural_predictions(model, validation_loader)
        score = f1_score(val.label, pred, average='macro')
        history.append({'epoch': epoch, 'training_loss': loss_sum / len(train), 'validation_macro_f1': score})
        print(f'Epoch {epoch}: loss={loss_sum / len(train):.4f}, validation macro F1={score:.4f}', flush=True)
        if score > best:
            best, stale = score, 0
            torch.save(model.state_dict(), OUT / 'neural.pt')
        else:
            stale += 1
        if stale >= 3:
            break
    neural_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(OUT / 'neural.pt', weights_only=True))
    rows.append(report('neural', 'validation', val, neural_predictions(model, validation_loader)))
    pd.DataFrame(history).to_csv(OUT / 'learning_history.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([h['epoch'] for h in history], [h['training_loss'] for h in history])
    axes[0].set(xlabel='Epoch', ylabel='Training loss')
    axes[1].plot([h['epoch'] for h in history], [h['validation_macro_f1'] for h in history])
    axes[1].set(xlabel='Epoch', ylabel='Validation macro F1')
    fig.tight_layout()
    fig.savefig(OUT / 'learning_curves.png', dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'validation_comparison.csv', index=False)
    import platform
    save_json(OUT / 'training.json', {'seed': SEED, 'device': 'cpu', 'platform': platform.platform(),
              'python': platform.python_version(), 'torch': torch.__version__, 'maximum_epochs': epochs,
              'selected_epoch': max(history, key=lambda h: h['validation_macro_f1'])['epoch'],
              'linear_training_seconds': linear_seconds, 'neural_training_seconds': neural_seconds,
              'total_seconds': time.perf_counter() - started})
    print(pd.DataFrame(rows).to_string(index=False))


def load_neural():
    vocab = json.loads((OUT / 'vocabulary.json').read_text())
    model = NeuralClassifier(len(vocab) + 2)
    model.load_state_dict(torch.load(OUT / 'neural.pt', map_location='cpu', weights_only=True))
    return model, vocab


def evaluate():
    if not (OUT / 'training.json').exists():
        raise SystemExit('Train both models successfully before final evaluation.')
    if (OUT / 'test').exists():
        raise SystemExit('Final test results already exist in outputs/test.')
    frame = pd.read_csv(DATA / 'raw_test.csv', header=None, names=['label', 'title', 'description'])
    frame, audit = clean(frame)
    # Apply the same predeclared conflict/duplicate policy to test data.
    save_json(OUT / 'test_cleaning.json', audit)
    rows = []
    for name in ['dummy', 'linear']:
        model = joblib.load(OUT / f'{name}.joblib')
        rows.append(report(name, 'test', frame, model.predict(frame.text)))
    model, vocab = load_neural()
    rows.append(report('neural', 'test', frame, neural_predictions(model, loader(frame, vocab))))
    pd.DataFrame(rows).to_csv(OUT / 'test_comparison.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    train = sub.add_parser('train')
    train.add_argument('--epochs', type=int, default=12)
    sub.add_parser('evaluate')
    predict = sub.add_parser('predict')
    predict.add_argument('--text', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'train':
        train_models(args.epochs)
    elif args.command == 'evaluate':
        seed_all()
        evaluate()
    else:
        if not args.text.strip():
            raise SystemExit('Enter a non-empty news text.')
        linear = joblib.load(OUT / 'linear.joblib')
        model, vocab = load_neural()
        frame = pd.DataFrame({'text': [args.text], 'label': [1]})
        print('Linear:', NAMES[int(linear.predict(frame.text)[0]) - 1])
        print('Neural:', NAMES[int(neural_predictions(model, loader(frame, vocab))[0]) - 1])


if __name__ == '__main__':
    main()
