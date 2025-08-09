import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class WeightedL1Loss(_Loss):
    def __init__(self, weights=None):
        super().__init__()

    def forward(self, y_pred, y_true):
        ndim = y_pred.ndim
        dims = list(range(1, ndim))
        loss = F.l1_loss(y_pred, y_true, reduction="none")
        loss = torch.mean(loss, dim=dims)
        weights = torch.mean(torch.abs(y_true), dim=dims)

        loss = torch.sum(loss * weights) / torch.sum(weights)

        return loss


class L1MatchLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        l1_true = torch.mean(torch.abs(y_true), dim=-1)
        l1_pred = torch.mean(torch.abs(y_pred), dim=-1)
        loss = torch.mean(torch.abs(l1_pred - l1_true))

        return loss


class DecibelMatchLoss(_Loss):
    def __init__(
        self,
        eps=1e-6,
        adaptive=True,
        min_weight=0.1,
        max_weight=1.0,
        min_db=-60,
        max_below_true=12,
    ):
        super().__init__()

        self.eps = eps
        self.adaptive = adaptive
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_db = min_db
        self.max_below_true = max_below_true

    def compute_weight(self, db_pred, db_true):
        with torch.no_grad():
            db_true_to_floor = torch.minimum(
                db_true - self.min_db,
                torch.tensor(self.max_below_true, device=db_true.device),
            )
            db_pred_to_true = db_true - db_pred

            ratio = torch.clamp(db_pred_to_true / db_true_to_floor, 0.0, 1.0)

            weight = self.min_weight + (self.max_weight - self.min_weight) * ratio

            # the closer db_pred is to db_true, the lower the weight

            weight = torch.where(
                (db_pred > db_true) | (db_true < self.min_db),
                torch.tensor(self.min_weight),
                weight,
            )

            assert torch.all(weight >= self.min_weight)
            assert torch.all(weight <= self.max_weight)

        return weight

    def forward(self, y_pred, y_true):
        out_dict = {}

        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        db_true = 10.0 * torch.log10(
            self.eps + torch.mean(torch.square(torch.abs(y_true)), dim=-1)
        )
        db_pred = 10.0 * torch.log10(
            self.eps + torch.mean(torch.square(torch.abs(y_pred)), dim=-1)
        )
        if self.adaptive:
            weights = self.compute_weight(db_pred, db_true)
        else:
            weights = 1.0

        diff = torch.abs(db_pred - db_true)

        loss = torch.mean(weights * diff)

        with torch.no_grad():
            out_dict["diff"] = diff.mean()
            if self.adaptive:
                out_dict["weights"] = weights.mean()

            out_dict["db_pred"] = db_pred.mean()

        out_dict["loss"] = loss

        return out_dict


class L1SNRLossIgnoreSilence(_Loss):
    def __init__(self, eps=1e-3, dbthresh=-20, dbthresh_step=20):
        super().__init__()
        self.eps = torch.tensor(eps)
        self.dbthresh = dbthresh
        self.dbthresh_step = dbthresh_step

    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        l1_error = torch.mean(torch.abs(y_pred - y_true), dim=-1)
        l1_true = torch.mean(torch.abs(y_true), dim=-1)

        snr = 20.0 * torch.log10((l1_true + self.eps) / (l1_error + self.eps))

        db = 10.0 * torch.log10(torch.mean(torch.square(y_true), dim=-1) + 1e-6)

        if torch.sum(db > self.dbthresh) == 0:
            if torch.sum(db > self.dbthresh - self.dbthresh_step) == 0:
                return -torch.mean(snr)
            else:
                return -torch.mean(snr[db > self.dbthresh - self.dbthresh_step])

        return -torch.mean(snr[db > self.dbthresh])


class L1SNRDecibelMatchLoss(_Loss):
    def __init__(self, db_weight=0.1, l1snr_eps=1e-3, dbeps=1e-3):
        super().__init__()
        # Use TimeDomainLoss for L1SNR component
        from banda.losses.time_domain_loss import TimeDomainLoss
        self.l1snr = TimeDomainLoss() # TimeDomainLoss uses calculate_l1_snr internally
        self.decibel_match = DecibelMatchLoss(dbeps)
        self.db_weight = db_weight

    def forward(self, y_pred, y_true):
        return self.l1snr(y_pred, y_true) + self.db_weight * self.decibel_match(y_pred, y_true)["loss"]