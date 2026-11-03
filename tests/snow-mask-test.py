"""Regression tests for snow/ice masking."""

import ast
import unittest
from pathlib import Path

import numpy as np
from skimage import morphology


def load_mask_functions():
    """Load the mask functions without importing optional GDAL dependencies."""
    source_path = Path(__file__).parent.parent / 'coastsat' / 'SDS_preprocess.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    function_names = {'create_cloud_mask', 'create_snow_mask',
                      'create_sentinel_snow_mask'}
    constant_names = {'S2_NDSI_THRESHOLD', 'S2_MIN_GREEN_REFLECTANCE'}
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in constant_names
                for target in node.targets):
            selected_nodes.append(node)
    namespace = {'np': np, 'morphology': morphology}
    module = ast.Module(body=selected_nodes, type_ignores=[])
    exec(compile(module, str(source_path), 'exec'), namespace)
    return namespace


class TestSnowMask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        functions = load_mask_functions()
        cls.create_cloud_mask = staticmethod(functions['create_cloud_mask'])
        cls.create_snow_mask = staticmethod(functions['create_snow_mask'])
        cls.create_sentinel_snow_mask = staticmethod(functions['create_sentinel_snow_mask'])

    def test_landsat_qa_pixel_snow_bit_is_masked(self):
        qa = np.zeros((20, 20), dtype=np.uint16)
        qa[4, 7] = 1 << 5

        for cloud_mask_issue in (False, True):
            with self.subTest(cloud_mask_issue=cloud_mask_issue):
                mask = self.create_cloud_mask(qa, 'L8', cloud_mask_issue)
                self.assertTrue(mask[4, 7])
                self.assertEqual(mask.sum(), 1)

    def test_sentinel_l1c_snow_mask_band_is_masked(self):
        qa = np.zeros((20, 20), dtype=np.uint16)
        snow = np.zeros((20, 20), dtype=np.uint8)
        snow[3, 8] = 1

        mask = self.create_snow_mask(qa, 'S2', snow)

        self.assertTrue(mask[3, 8])
        self.assertEqual(mask.sum(), 1)

    def test_legacy_sentinel_qa60_without_snow_band_has_no_snow_mask(self):
        qa = np.zeros((20, 20), dtype=np.uint16)

        mask = self.create_snow_mask(qa, 'S2')

        self.assertFalse(mask.any())

    def test_nome_alaska_l1c_snow_mask_is_close_to_scl_reference(self):
        fixture_path = Path(__file__).parent / 'examples' / 'nome-l1c-validation.npz'
        with np.load(fixture_path, allow_pickle=False) as fixture:
            green = fixture['green_20m'].astype(np.float32) / 10000
            swir1 = fixture['swir1_20m'].astype(np.float32) / 10000
            scl = fixture['scl_20m'].astype(np.uint8)

        valid = (scl != 0) & (scl != 1)
        snow_mask = self.create_sentinel_snow_mask(green, swir1) & valid
        scl_snow = (scl == 11) & valid
        intersection = np.logical_and(snow_mask, scl_snow).sum()
        union = np.logical_or(snow_mask, scl_snow).sum()
        iou = intersection / union
        ndsi_coverage = snow_mask.sum() / valid.sum() * 100

        self.assertAlmostEqual(ndsi_coverage, 22.4, delta=0.2)
        self.assertGreater(iou, 0.75)

    def test_existing_landsat_cloud_mask_is_preserved(self):
        qa = np.zeros((20, 20), dtype=np.uint16)
        qa[4:12, 5:13] = 1 << 3

        mask = self.create_cloud_mask(qa, 'L8', False)

        self.assertTrue(mask[6, 6])
        self.assertFalse(mask[0, 0])

    def test_existing_sentinel_qa60_cloud_mask_is_preserved(self):
        qa = np.zeros((20, 20), dtype=np.uint16)
        qa[4:12, 5:13] = 1 << 10

        mask = self.create_cloud_mask(qa, 'S2', False)

        self.assertTrue(mask[6, 6])
        self.assertFalse(mask[0, 0])


if __name__ == '__main__':
    unittest.main()
