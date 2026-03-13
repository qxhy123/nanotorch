import numpy as np

from nanotorch.nn.metrics import RecommenderMetrics, auc_score, log_loss
from nanotorch.tensor import Tensor


def test_auc_and_log_loss_accept_tensor_inputs():
    predictions = Tensor(np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32))
    targets = Tensor(np.array([1, 0, 1, 0], dtype=np.float32))

    auc = auc_score(predictions, targets)
    loss = log_loss(predictions, targets)

    assert 0.99 <= auc <= 1.0
    assert loss >= 0.0


def test_recommender_metrics_update_snapshots_batches():
    metrics = RecommenderMetrics(ks=[1, 2])
    predictions = Tensor(np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32))
    targets = Tensor(np.array([[1, 0], [0, 1]], dtype=np.float32))

    metrics.update(predictions, targets)

    predictions.data[:] = 0.0
    targets.data[:] = 0.0

    results = metrics.compute()

    assert results["hit@1"] == 1.0
    assert results["recall@2"] == 1.0
    assert metrics._count == 2
