"""Verify delay alignment, threshold boundaries and forecast information use."""
import unittest

import numpy as np

from setar_tar_usd_vnd import SETARModel, common_metrics, fit_setar, forecast_one_step
from setar_tar_usd_vnd import select_models
from setar_evaluation_usd_vnd import full_metrics, interval_scores
import pandas as pd


class SETARTests(unittest.TestCase):
    def test_threshold_delay_can_exceed_ar_order_and_equality_is_low(self):
        model = SETARModel(2, 4, 1.0, np.array([1., 2., 3.]),
                           np.array([-1., -2., -3.]), 4, 0., 10, 5, 5)
        history = np.array([1., 2., 3., 4.])
        observed = np.array([5., 6., 7.])
        pred, regime, z = forecast_one_step(model, history, observed)
        np.testing.assert_array_equal(z, [1., 2., 3.])
        np.testing.assert_array_equal(regime, ['Low', 'High', 'High'])
        np.testing.assert_allclose(pred, [1+2*4+3*3, -1-2*5-3*4, -1-2*6-3*5])

    def test_current_and_future_targets_cannot_change_current_forecast(self):
        model = SETARModel(2, 3, 0., np.array([.1, .2, .3]),
                           np.array([-.1, -.2, -.3]), 3, 1., 100, 50, 50)
        history = np.array([1., -2., 3., -4.])
        observed = np.array([.5, -.6, .7, -.8, .9])
        expected = forecast_one_step(model, history, observed)[0]
        for cutoff in range(len(observed)):
            changed = observed.copy()
            changed[cutoff:] = 10000.
            actual = forecast_one_step(model, history, changed)[0]
            np.testing.assert_allclose(actual[:cutoff+1], expected[:cutoff+1])

    def test_regime_fit_matches_independent_block_design(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=1200)
        model = fit_setar(y, 2, 3, 0., start=4)
        t = np.arange(4, len(y))
        X = np.column_stack([np.ones(len(t)), y[t-1], y[t-2]])
        low = y[t-3] <= 0
        block = np.column_stack([X*low[:, None], X*(~low)[:, None]])
        joint, *_ = np.linalg.lstsq(block, y[t], rcond=None)
        np.testing.assert_allclose(np.r_[model.beta_low, model.beta_high], joint, atol=1e-12)
        np.testing.assert_allclose(model.rss, np.sum((y[t]-block@joint)**2))

    def test_insufficient_regime_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Too few observations'):
            fit_setar(np.arange(100, dtype=float), 2, 1, -1, start=4)

    def test_common_metrics_use_declared_naive_scales(self):
        actual = np.array([10., 12.])
        predicted = np.array([9., 13.])
        result = common_metrics(actual, predicted, mase_scale=.5,
                                naive_rmse=2., previous=np.array([8., 12.]),
                                percentage_metrics=True)
        self.assertAlmostEqual(result['MAE'], 1.)
        self.assertAlmostEqual(result['RMSE'], 1.)
        self.assertAlmostEqual(result['MASE'], 2.)
        self.assertAlmostEqual(result['Relative_RMSE_vs_Naive'], .5)
        self.assertAlmostEqual(result['Direction_accuracy'], .5)
        self.assertAlmostEqual(result['MAPE_percent'], 100*((1/10+1/12)/2))

    def test_model_selection_cannot_depend_on_test_values(self):
        rng = np.random.default_rng(731)
        splits = {name: pd.DataFrame({'log_return': rng.normal(size=n)})
                  for name, n in [('train', 400), ('validation', 100), ('test', 20)]}
        first = select_models(splits, orders=[1, 2], delays=[1, 2])[0]
        splits['test']['log_return'] = 1e8
        second = select_models(splits, orders=[1, 2], delays=[1, 2])[0]
        for name in ('AR', 'SETAR'):
            self.assertEqual(first[name].p, second[name].p)
        self.assertEqual(first['SETAR'].d, second['SETAR'].d)
        self.assertEqual(first['SETAR'].threshold, second['SETAR'].threshold)
        np.testing.assert_array_equal(first['SETAR'].beta_low, second['SETAR'].beta_low)

    def test_direction_counts_handle_flat_and_zero_return_mape(self):
        result = full_metrics([0., 1., -1.], [0., 1., 1.], [0., 1., 0.], [0., 0., 0.])
        self.assertAlmostEqual(result['DA_all_percent'], 200/3)
        self.assertAlmostEqual(result['DA_nonzero_actual_percent'], 50.)
        self.assertEqual(result['Actual_flat_N'], 1)
        self.assertTrue(np.isnan(result['MAPE_percent']))

    def test_interval_score_penalizes_missed_observations(self):
        result = interval_scores([0., 3.], [-1., -1.], [1., 1.])
        self.assertEqual(result['PI_coverage_percent'], 50.)
        self.assertEqual(result['Average_PI_width'], 2.)
        self.assertAlmostEqual(result['Mean_interval_score'], 42.)


if __name__ == '__main__':
    unittest.main()

