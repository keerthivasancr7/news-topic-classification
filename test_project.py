import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ml_project", Path(__file__).resolve().parents[1] / "ML project.py"
)
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)
clean = project.clean
NeuralClassifier = project.NeuralClassifier
TextData = project.TextData


class ProjectTests(unittest.TestCase):
    def test_conflicts_duplicates_and_empty(self):
        frame = pd.DataFrame({
            'label': [1, 1, 2, 3, 4],
            'title': ['same', 'SAME', 'conflict', 'conflict', ''],
            'description': ['text', 'text', 'x', 'x', '']})
        result, audit = clean(frame)
        self.assertEqual(result.label.tolist(), [1])
        self.assertEqual(audit['conflicting_texts'], 1)
        self.assertEqual(audit['empty_texts'], 1)

    def test_padding_does_not_change_output(self):
        torch.manual_seed(42)
        model = NeuralClassifier(10).eval()
        with torch.no_grad():
            a = model(torch.tensor([[2, 3]]))
            b = model(torch.tensor([[2, 3, 0, 0]]))
        torch.testing.assert_close(a, b)

    def test_unknown_and_empty_tokens(self):
        rows = TextData(['known unseen', '!!!'], [1, 4], {'known': 2})
        self.assertEqual(rows[0], ([2, 1], 0))
        self.assertEqual(rows[1], ([1], 3))

    def test_groups_remain_separate_and_reproducible(self):
        groups = np.repeat(np.arange(40), 2)
        labels = np.repeat(np.tile([1, 2, 3, 4], 10), 2)
        def split():
            return next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                        .split(groups, labels, groups))
        tr, va = split()
        self.assertTrue(set(groups[tr]).isdisjoint(groups[va]))
        np.testing.assert_array_equal(tr, split()[0])
        self.assertEqual(set(labels[va]), {1, 2, 3, 4})


class WorkflowTests(unittest.TestCase):
    def test_saved_models_and_evaluation(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        # Synthetic fixtures exercise the workflow only, never project results.
        rows = []
        words = ['world diplomacy', 'sports football', 'business markets', 'science computers']
        for label, word in enumerate(words, 1):
            for i in range(15):
                rows.append({'label': label, 'text': f'{word} sample {i}'})
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            data, out = Path(directory) / 'data', Path(directory) / 'outputs'
            data.mkdir()
            frame.to_csv(data / 'train.csv', index=False)
            frame.iloc[::3].to_csv(data / 'validation.csv', index=False)
            raw_test = pd.DataFrame({'label': [1, 2, 3, 4], 'title': words,
                                     'description': ['heldout report'] * 4})
            raw_test.to_csv(data / 'raw_test.csv', header=False, index=False)
            with patch.object(project, 'DATA', data), patch.object(project, 'OUT', out):
                project.train_models(1)
                project.evaluate()
                self.assertTrue((out / 'test_comparison.csv').exists())
                self.assertTrue((out / 'learning_curves.png').exists())
                self.assertEqual(len(pd.read_csv(out / 'test_comparison.csv')), 3)
                with self.assertRaises(SystemExit):
                    project.train_models(1)
                with self.assertRaises(SystemExit):
                    project.evaluate()


    def test_prepare_excludes_test_overlap_and_keeps_groups(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        rows = []
        for label in range(1, 5):
            for i in range(20):
                rows.append([label, f'topic {label} story {i}', f'description {i}'])
        rows.append([1, 'topic 1 story 2', 'another description'])
        with tempfile.TemporaryDirectory() as directory:
            data, out = Path(directory) / 'data', Path(directory) / 'outputs'
            data.mkdir()
            pd.DataFrame(rows).to_csv(data / 'raw_train.csv', index=False, header=False)
            pd.DataFrame([[4, 'TOPIC 1 STORY 0', 'different text']]).to_csv(
                data / 'raw_test.csv', index=False, header=False)
            with patch.object(project, 'DATA', data), patch.object(project, 'OUT', out):
                project.prepare()
            train = pd.read_csv(data / 'train.csv')
            val = pd.read_csv(data / 'validation.csv')
            self.assertTrue(set(train.group).isdisjoint(val.group))
            self.assertNotIn('topic 1 story 0', set(train.group) | set(val.group))
            self.assertEqual(len(train) + len(val), 80)


if __name__ == '__main__':
    unittest.main()
