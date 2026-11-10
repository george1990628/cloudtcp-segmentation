#!/usr/bin/env python3
"""Audit the existing dataset or score an existing run's saved predictions.

No training or model download. Dependencies: numpy, rasterio, matplotlib.
The saved prediction TIFFs, not screenshot pixels, are used for exact scoring.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import rasterio

from baseline_dataset_core import extract_dataset, inspect_dataset, metrics_from_confusion, sha256
from compare_per_class_iou import CLASSES, REF_FIELDS, make_reference, load_metrics, write_csv


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def get_bundle(dataset, cache):
    dataset = Path(dataset)
    root = extract_dataset(dataset, cache) if dataset.is_file() else dataset
    bundle = inspect_dataset(root, input_bands=(1, 2, 3))
    if bundle['names'] != list(CLASSES.values()):
        raise ValueError('Class names/order differ from the original five-output baseline.')
    return bundle


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output is not empty: {path}. Choose a new folder to preserve earlier results.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def pixel_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def audit_dataset(bundle, output, dataset=None):
    output = new_output(output)
    audit = dict(bundle['audit'])
    audit['measurement_kind'] = 'dataset_audit; not trained-model performance'
    audit['source_archive_sha256'] = sha256(dataset) if dataset and Path(dataset).is_file() else None
    hashes = {}
    for sample in bundle['samples']:
        hashes.setdefault(pixel_hash(sample['image']), []).append(sample['row'])
    audit['duplicate_rgb_pixels_across_splits'] = [
        [r['chip'] for r in rows] for rows in hashes.values()
        if len({r['split'] for r in rows}) > 1
    ]
    touching, minimum_gap = [], math.inf
    for a in bundle['splits']['train']:
        for b in bundle['splits']['val']:
            aa, bb = a['bounds'], b['bounds']
            dx = max(0., aa[0] - bb[2], bb[0] - aa[2])
            dy = max(0., aa[1] - bb[3], bb[1] - aa[3])
            gap = math.hypot(dx, dy)
            minimum_gap = min(minimum_gap, gap)
            if gap < 1e-4:
                touching.append([a['row']['chip'], b['row']['chip']])
    audit['touching_train_val_footprints'] = touching
    audit['minimum_train_val_footprint_gap_m'] = minimum_gap
    audit['geographic_caveat'] = ('No overlap/group collision does not establish independent geographic '
                                  'generalization; neighboring chips may share local appearance.')
    spacings = [[abs(s['profile']['transform'].a), abs(s['profile']['transform'].e)]
                for s in bundle['samples']]
    audit['exported_pixel_spacing_m_range'] = {
        'x': [min(p[0] for p in spacings), max(p[0] for p in spacings)],
        'y': [min(p[1] for p in spacings), max(p[1] for p in spacings)]}
    audit['native_acquisition_gsd'] = 'unknown; exported pixel spacing is not verified sensor GSD'
    audit['input_bands_note'] = 'Bands 1–3 treated as RGB, matching the original baseline; band 4 unused.'
    audit['validity_rule'] = 'Ignore mask 255 and pixels invalid in any selected image band, as in the baseline.'
    audit['checks_passed'] = [
        'unique chip IDs and existing image/mask pairs', 'allowed label IDs 0–4 and 255',
        'image/mask dimensions, CRS and affine alignment', 'uint8 imagery and single-band integer masks',
        'nonempty train/validation sets and all foreground classes present in training',
        'no shared train/validation group IDs', 'no positive-area train/validation footprint overlap']
    if not audit['duplicate_rgb_pixels_across_splits']:
        audit['checks_passed'].append('no identical RGB pixel arrays across train/validation')
    write_json(output/'data_audit.json', audit)
    write_csv(output/'class_distribution.csv', tuple(audit['class_report'][0]), audit['class_report'])
    write_csv(output/'validation_reference.csv', REF_FIELDS, make_reference(bundle['root']))
    chip_rows = []
    for sample in bundle['samples']:
        row = sample['row']
        values = np.bincount(sample['mask'][sample['mask'] != 255], minlength=5)
        chip_rows.append({'chip': row['chip'], 'split': row['split'],
                          'label_status': row.get('label_status', 'unknown'),
                          'group': row.get('group', ''), 'valid_pixels': int(values.sum()),
                          'ignored_pixels': int((sample['mask'] == 255).sum()),
                          **{name+'_pixels': int(values[i]) for i, name in CLASSES.items()}})
    write_csv(output/'per_chip_support.csv', tuple(chip_rows[0]), chip_rows)
    return audit


def accumulate(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError('Prediction/mask shape mismatch.')
    valid = truth != 255
    if not set(np.unique(prediction[valid]).tolist()).issubset(set(range(5))):
        raise ValueError('Prediction must contain a class ID 0–4 at every evaluable pixel; 255 cannot hide errors.')
    return np.bincount((truth[valid] * 5 + prediction[valid].astype(np.int64)).ravel(),
                       minlength=25).reshape(5, 5)


def score_predictions(bundle, predictions):
    predictions = Path(predictions)
    confusion = np.zeros((5, 5), dtype=np.int64)
    chip_rows, prediction_refs = [], []
    for sample in bundle['splits']['val']:
        chip = sample['row']['chip']
        path = predictions/(chip+'.tif')
        if not path.is_file():
            raise ValueError(f'Missing saved prediction: {path}')
        with rasterio.open(path) as src:
            if src.count != 1 or not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
                raise ValueError(f'{chip}: prediction must be a single-band integer TIFF.')
            if (src.height, src.width) != sample['mask'].shape or src.crs != sample['profile']['crs']:
                raise ValueError(f'{chip}: prediction shape or CRS does not match the image.')
            if not src.transform.almost_equals(sample['profile']['transform']):
                raise ValueError(f'{chip}: prediction grid does not match the image.')
            prediction = src.read(1)
        matrix = accumulate(sample['mask'], prediction)
        confusion += matrix
        scores = metrics_from_confusion(matrix, bundle['names'])
        chip_rows.append({'chip': chip, 'foreground_mIoU': scores['foreground_miou'],
                          **scores['iou']})
        prediction_refs.append({'chip': chip, 'prediction_sha256': sha256(path)})
    return confusion, chip_rows, prediction_refs


def write_scores(bundle, confusion, chip_rows, prediction_refs, output, metadata):
    output = new_output(output)
    scores = metrics_from_confusion(confusion, bundle['names'])
    write_json(output/'metrics.json', {**metadata, **scores})
    rows = []
    for i, name in CLASSES.items():
        tp, actual, predicted = int(confusion[i, i]), int(confusion[i].sum()), int(confusion[:, i].sum())
        rows.append({'class': name, 'IoU': scores['iou'][name], 'F1': scores['f1'][name],
                     'precision': tp/predicted if predicted else None,
                     'recall': tp/actual if actual else None,
                     'label_pixels': actual, 'predicted_pixels': predicted})
    write_csv(output/'per_class_metrics.csv', tuple(rows[0]), rows)
    write_csv(output/'per_chip_metrics.csv', tuple(chip_rows[0]), chip_rows)
    write_csv(output/'prediction_reference.csv', tuple(prediction_refs[0]), prediction_refs)
    write_csv(output/'validation_reference.csv', REF_FIELDS, make_reference(bundle['root']))
    matrix_rows = [{'actual_class': name, **{pred: int(confusion[i, j])
                    for j, pred in CLASSES.items()}} for i, name in CLASSES.items()]
    write_csv(output/'confusion_matrix.csv', tuple(matrix_rows[0]), matrix_rows)
    return scores


def evaluate_run(bundle, run, output):
    run, output = Path(run), Path(output)
    original_audit = run/'data_audit.json'
    if not original_audit.is_file():
        raise ValueError('Run is missing data_audit.json; its dataset provenance cannot be checked.')
    recorded_audit = json.loads(original_audit.read_text())
    if recorded_audit.get('dataset_fingerprint') != bundle['audit']['dataset_fingerprint']:
        raise ValueError('Dataset differs from the saved run. Use that run\'s original dataset; '
                         'changed-label evaluation requires regenerating predictions/provenance deliberately.')
    if recorded_audit.get('input_bands') != [1, 2, 3]:
        raise ValueError('Saved run used different image bands; its valid-pixel rule needs a matching evaluator.')
    old_metrics = json.loads((run/'metrics.json').read_text()) if (run/'metrics.json').is_file() else {}
    model_config = run/'best_model/config.json'
    if model_config.is_file():
        config = json.loads(model_config.read_text())
        if config.get('id2label') != {str(k): v for k, v in CLASSES.items()}:
            raise ValueError('Saved model class mapping does not match the dataset.')
    confusion, chip_rows, refs = score_predictions(bundle, run/'predictions')
    checkpoint_files = {}
    if (run/'best_model').is_dir():
        checkpoint_files = {str(p.relative_to(run)): sha256(p)
                            for p in sorted((run/'best_model').rglob('*')) if p.is_file()}
    metadata = {'measurement_kind': 'recomputed from existing saved prediction TIFFs; no training',
                'source_run_name': run.name, 'dataset_fingerprint': bundle['audit']['dataset_fingerprint'],
                'completed_epochs': old_metrics.get('completed_epochs'),
                'best_epoch': old_metrics.get('best_epoch'),
                'label_status': bundle['audit']['label_status'],
                'checkpoint_file_hashes': checkpoint_files,
                'metric_definition': 'global pixel confusion matrix; mean of foreground class IoUs',
                'evaluation_caveat': 'Agreement with supplied draft masks, not verified real-world accuracy.'}
    scores = write_scores(bundle, confusion, chip_rows, refs, output, metadata)
    original_dir = output/'original_run_files'
    original_dir.mkdir()
    for name in ['metrics.json', 'per_class_metrics.csv', 'per_chip_metrics.csv', 'data_audit.json',
                 'training_config.json', 'history.csv', 'history.json', 'TRAINING_NOTE.md',
                 'learning_curves.png', 'qualitative_results.png']:
        if (run/name).is_file():
            shutil.copy2(run/name, original_dir/name)
    if (run/'per_class_metrics.csv').is_file():
        old = load_metrics(run/'per_class_metrics.csv')
        rows = [{'class': name, 'original_logged_IoU': old.get(name),
                 'saved_prediction_IoU': scores['iou'][name],
                 'difference': scores['iou'][name]-old[name] if name in old else None}
                for name in bundle['names']]
        write_csv(output/'original_vs_saved_prediction_scores.csv', tuple(rows[0]), rows)
    (output/'EVALUATION_NOTE.md').write_text(
        '# Existing baseline evaluation\n\n'
        f"Scored {len(bundle['splits']['val'])} existing validation prediction TIFFs. No retraining.\n\n"
        f"Foreground mIoU against the supplied labels: **{scores['foreground_miou']:.6f}**.\n\n"
        'The original run files are preserved separately. Small differences from the logged '
        'validation scores may arise because training-time validation and exported prediction '
        'TIFFs used different inference batch sizes. Larger differences require investigation. '
        'Use the same saved-prediction scoring procedure for subsequent comparisons.\n\n'
        'If validation labels are corrected, score predictions from both checkpoints on the same '
        'corrected labels and record a new validation reference. The guarded original-run command '
        'intentionally refuses an unexplained dataset change.\n')
    return scores


def run_controls(bundle, output):
    output = new_output(output)
    results = {}
    with tempfile.TemporaryDirectory(prefix='cloudtcp_controls_') as temporary:
        temp = Path(temporary)
        for control in ('all_background', 'perfect_copy_of_labels'):
            folder = temp/control
            folder.mkdir()
            for sample in bundle['splits']['val']:
                prediction = np.zeros(sample['mask'].shape, dtype=np.uint8)
                if control == 'perfect_copy_of_labels':
                    prediction = sample['mask'].astype(np.uint8)
                prediction[sample['mask'] == 255] = 255
                profile = dict(sample['profile'])
                profile.update(driver='GTiff', count=1, dtype='uint8', nodata=255, compress='deflate')
                profile.pop('photometric', None)
                with rasterio.open(folder/(sample['row']['chip']+'.tif'), 'w', **profile) as dst:
                    dst.write(prediction, 1)
            matrix, chips, refs = score_predictions(bundle, folder)
            result = write_scores(bundle, matrix, chips, refs, output/control,
                                  {'measurement_kind': 'sanity-check control; NOT a trained model',
                                   'control': control})
            expected = 0. if control == 'all_background' else 1.
            if abs(result['foreground_miou'] - expected) > 1e-12:
                raise RuntimeError(f'Scoring check failed: {control}')
            results[control] = {'foreground_mIoU': result['foreground_miou'],
                                'pixel_accuracy': result['pixel_accuracy'], 'expected_foreground_mIoU': expected}
    write_json(output/'CONTROL_RESULTS.json', {
        'purpose': 'Verify scoring on all 19 actual validation grids. These are not trained-model results.',
        'controls': results})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'controls', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', required=True, help='Dataset ZIP or dataset root folder')
        command.add_argument('--output', required=True, help='New output folder')
        command.add_argument('--cache', default='.cloudtcp_dataset_cache')
        if name == 'evaluate':
            command.add_argument('--run', required=True, help='Existing run folder with prediction TIFFs')
    args = parser.parse_args()
    try:
        bundle = get_bundle(args.dataset, args.cache)
        if args.command == 'audit':
            result = audit_dataset(bundle, args.output, args.dataset)
            print(json.dumps({'status': 'passed', 'n_chips': result['n_chips'],
                              'foreground_train_pixels': result['class_counts']['train'][1:],
                              'minimum_train_val_footprint_gap_m': result['minimum_train_val_footprint_gap_m']}, indent=2))
        elif args.command == 'controls':
            print(json.dumps(run_controls(bundle, args.output), indent=2))
        else:
            result = evaluate_run(bundle, args.run, args.output)
            print(json.dumps({'foreground_mIoU': result['foreground_miou'], 'IoU': result['iou']}, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
