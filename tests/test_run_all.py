from pathlib import Path

from scripts import run_all


def test_run_all_orders_both_datasets_and_evaluations(tmp_path, monkeypatch):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/default.yaml').write_text('model:\n  path: models/sd35-medium\n')
    model = tmp_path / 'models/sd35-medium'
    model.mkdir(parents=True)
    (model / 'model_index.json').write_text('{}')
    images = tmp_path / 'data/MSD/Task10_Colon/imagesTr'
    labels = tmp_path / 'data/MSD/Task10_Colon/labelsTr'
    monkeypatch.setattr(run_all, 'prepare_msd', lambda root: (images, labels))
    monkeypatch.setattr(run_all, 'msi_pairs', lambda root: [('10001', Path('image.nii.gz'), Path('label.nii.gz'))])
    calls = []

    def capture(command, **kwargs):
        calls.append((Path(command[1]).name, command[2:], kwargs))

    run_all.run_all(tmp_path, runner=capture)
    assert [name for name, _, _ in calls] == [
        'run_inference.py', 'evaluate_msd.py', 'run_inference.py', 'evaluate_msd.py'
    ]
    assert all(kwargs['check'] and kwargs['cwd'] == tmp_path for _, _, kwargs in calls)
    assert '--resume' in calls[0][1] and '--resume' in calls[2][1]


def test_run_all_downloads_missing_model_first(tmp_path, monkeypatch):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/default.yaml').write_text('model:\n  path: models/sd35-medium\n')
    monkeypatch.setattr(run_all, 'prepare_msd', lambda root: (Path('imagesTr'), Path('labelsTr')))
    monkeypatch.setattr(run_all, 'msi_pairs', lambda root: [('10001', Path('image'), Path('label'))])
    calls = []
    run_all.run_all(tmp_path, runner=lambda command, **kwargs: calls.append(Path(command[1]).name))
    assert calls[0] == 'download_model.py'
