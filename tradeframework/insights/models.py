from tradeframework.api.insights import InsightGenerator
import quantutils.core.statistics as stats
import tradeframework.operations.utils as utils
import numpy as np
import tradeframework.operations.plot as plotter
import warnings


class ARIMAFit(InsightGenerator):
    """
    Fits an ARIMA regression model to the specified time series
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("order", None)  # Tuple of AR,I,MA
        self.opts.setdefault("series", "returns")

    def getInsight(self, derivative, display=True):
        warnings.filterwarnings("ignore")

        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        result = stats.ARIMAFit(ts=series, order=self.opts["order"], display=True)

        # if display:
        #    plotter.tsplot(result.resid, lags=30, title="Residuals")

        return result
