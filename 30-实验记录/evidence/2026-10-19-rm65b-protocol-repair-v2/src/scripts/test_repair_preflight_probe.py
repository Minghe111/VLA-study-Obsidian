import importlib.util
from pathlib import Path
import types
import unittest

import torch

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('repair_preflight_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tests(unittest.TestCase):
    def test_chunked_fp32_projection_matches_full(self):
        torch.manual_seed(4)
        x = torch.randn(5, 7).bfloat16()
        w = torch.randn(19, 7).bfloat16()
        labels = torch.tensor([0, 2, 18, 12, 6])
        expected = probe.logits_stats(torch.nn.functional.linear(x.float(), w.float()), labels)
        actual = probe.fp32_projection_stats(x, w, labels, chunk_size=4)
        self.assertEqual(actual['top1'], expected['top1'])
        self.assertEqual(actual['gold_rank'], expected['gold_rank'])
        torch.testing.assert_close(torch.tensor(actual['gold_ce']), torch.tensor(expected['gold_ce']), atol=1e-6, rtol=1e-6)

    def test_rank_is_optimistic_tie_rank(self):
        result = probe.logits_stats(torch.tensor([[2., 2., 1.], [0., 1., 3.]]), torch.tensor([1, 0]))
        self.assertEqual(result['gold_rank'], [1, 3])

    def test_gradient_norm_cosine_and_weighted_combination(self):
        values = probe.gradient_summary(['llm.layers.0.attn.q_proj.0.lora_A'],
            [torch.tensor([3., 4.])], [torch.tensor([-3., -4.])], .1)
        self.assertAlmostEqual(values['all']['ce_grad_norm'], 5.)
        self.assertAlmostEqual(values['all']['kv_grad_norm'], 5.)
        self.assertAlmostEqual(values['all']['cosine'], -1.)
        self.assertAlmostEqual(values['all']['weighted_combined_grad_norm'], 4.5)
        self.assertAlmostEqual(values['llm.layers.0']['weighted_kv_over_ce_norm'], .1)

    def test_zero_gradient_has_no_defined_cosine(self):
        values = probe.gradient_summary(['a'], [torch.zeros(2)], [torch.zeros(2)])
        self.assertIsNone(values['all']['cosine'])
        self.assertIsNone(values['all']['weighted_kv_over_ce_norm'])

    def test_token_regions_key_body_first_eos_from_exact_spans(self):
        text = '{"x":"a"}'
        pieces = [types.SimpleNamespace(id=i, begin=i, end=i+1) for i in range(len(text))]
        processor = types.SimpleNamespace(encode_as_immutable_proto=lambda t: types.SimpleNamespace(pieces=pieces), id_to_piece=lambda i: text[i])
        regions, origin = probe.token_regions(text, list(range(len(text))) + [99], types.SimpleNamespace(text_tokenizer=processor))
        self.assertEqual(origin, 'sentencepiece_immutable_proto')
        self.assertTrue(regions[0]['first'])
        self.assertEqual(regions[2]['region'], 'field_key')
        self.assertEqual(regions[6]['region'], 'body_or_syntax')
        self.assertTrue(regions[-1]['eos'])
        self.assertEqual(len(regions), len(text)+1)

    def test_fp32_projection_invalid_shape_rejected(self):
        with self.assertRaises(ValueError):
            probe.fp32_projection_stats(torch.zeros(2,3), torch.zeros(7,3), torch.tensor([1]), chunk_size=4)


if __name__ == '__main__':
    unittest.main()
