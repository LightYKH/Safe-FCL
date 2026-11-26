import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'evaluation/scripts'))
import evaluate_task
import release_utils
import validate_release
from release_utils import SAFE_FCL_METHOD, assert_profile, load, method_config_path
from materialize_code import materialize

class ReleaseTests(unittest.TestCase):
    def test_legacy_method_alias_assembles_the_canonical_method(self):
        archived_method = "archived_method"
        archived_digest = hashlib.sha256(archived_method.encode("ascii")).hexdigest()
        self.assertEqual(SAFE_FCL_METHOD, "safe_fcl")
        with patch.object(release_utils, "_LEGACY_METHOD_SHA256", archived_digest):
            for group in ("methods", "gates"):
                self.assertEqual(
                    method_config_path(archived_method, group),
                    method_config_path(SAFE_FCL_METHOD, group),
                )
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                current = materialize(SAFE_FCL_METHOD, root / "current", collection=True)
                archived = materialize(archived_method, root / "archived", collection=True)
                current_files = {path.relative_to(current): path.read_bytes()
                                 for path in current.rglob("*") if path.is_file()}
                archived_files = {path.relative_to(archived): path.read_bytes()
                                  for path in archived.rglob("*") if path.is_file()}
                self.assertEqual(archived_files, current_files)

    def test_nonselected_profiles_rejected(self):
        args=load(method_config_path(SAFE_FCL_METHOD))['training']
        assert_profile(args)
        for change in [dict(batch_size=2,gradient_accumulation_steps=8),dict(seq_length=256),dict(seq_length=1024),dict(batch_size=True)]:
            with self.assertRaises(ValueError):assert_profile(dict(args,**change))

    def test_order_and_task5_materialization(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'code'
            materialize(SAFE_FCL_METHOD,target,5,'order1')
            self.assertIn("'benign': 'risk5/task1_benign'",(target/'utils/continual_data_manifest.py').read_text())
            self.assertIn('_FrozenTask4Guard',(target/'utils/online_probe_guard.py').read_text())
            self.assertIn('batch_size=4',(target/'utils/online_experiment_contract.py').read_text())
            with self.assertRaises(FileExistsError):materialize(SAFE_FCL_METHOD,target)

    def test_all_method_profiles(self):
        for p in (ROOT/'configs/methods').glob('*.json'):assert_profile(load(p)['training'])

    def test_evaluation_preflight_leaves_output_available(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            output = root / 'evaluation'
            args = SimpleNamespace(
                base_model=root, adapter=root, data_dir=root, judge=root, manifest=manifest,
                output_dir=output, task=1, order='order2', preflight=True,
            )
            source_paths = []

            def check_materialized_code(_):
                source_paths.append(Path(sys.path[0]))
                self.assertTrue((source_paths[-1] / 'main_sft.py').is_file())

            with patch.object(evaluate_task, '_evaluate', side_effect=check_materialized_code):
                evaluate_task.evaluate(args)
                self.assertFalse(output.exists())
                self.assertFalse(source_paths[0].exists())
                args.preflight = False
                evaluate_task.evaluate(args)
                self.assertEqual(source_paths[1], output / 'code')
                self.assertTrue(source_paths[1].is_dir())

    def test_release_scan_detects_split_private_locators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            address = '.'.join(('192', '0', '2', '10'))
            prefix = '/'.join(('', 'home', ''))
            username = 'researcher'
            for name, text in (
                ('host.txt', address),
                ('path.py', f'ROOT = {prefix!r} + {username!r}'),
            ):
                (root / name).write_text(text, encoding='utf-8')
            with patch.object(validate_release, 'ROOT', root):
                result = validate_release.validate()
            self.assertFalse(result['passed'])
            self.assertCountEqual(result['problems'], [
                'Possible private path or host: host.txt',
                'Possible private path or host: path.py',
            ])

if __name__=='__main__':unittest.main()
