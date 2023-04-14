import pandas as pd
import numpy as np
from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
import quantutils.core.timeseries as tsUtils
from IPython.display import display as displayResult
import tradeframework.operations.plot as plotter
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


class TimeSeriesPlot(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("lags", 30)
        self.opts.setdefault("series", "returns")

    def getInsight(self, derivative, display=True):
        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        fig = plotter.tsplot(series, lags=self.opts["lags"], show=False)
        if display:
            displayResult(fig)
        return fig


class AutoCorrelationPlot(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("lags", 30)
        self.opts.setdefault("series", "returns")

    def getInsight(self, derivative, display=True):
        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        del self.opts["series"]
        acf_results = pd.DataFrame(tsUtils.autocorr(series), index=series.index)
        if display:
            ax = plotter.outlinePlot(title="AutoCorrelation Plot")
            plot_acf(x=series, ax=ax, **self.opts)
        return acf_results


# Moving AutoCorrelation Plot


class MACFPlot(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("lag", 1)
        self.opts.setdefault("window", 14)
        self.opts.setdefault("offset", int(self.opts["window"] / 2))
        self.opts.setdefault("series", "returns")

    def getInsight(self, derivative, display=True):
        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        del self.opts["series"]
        macf_results = pd.DataFrame(
            tsUtils.MACF(series.values, **self.opts), index=series.index
        )
        if display:
            feeds = []
            feeds.append(
                {
                    "data": macf_results,
                    "opts": {
                        "label": derivative.getName(),
                        "color": "red",
                        "marker": "o",
                        "linestyle": "None",
                    },
                }
            )
            plotter.basicPlot(
                title=f"Moving AutoCorrelation Plot: {derivative.getName()}",
                feeds=feeds,
            )

        return macf_results


# Regime identification


class MarkovRegimeFit(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("k_regimes", 2)
        self.opts.setdefault("order", 1)
        self.opts.setdefault("trend", "nc")
        self.opts.setdefault("switching_variance", True)

    def getInsight(self, derivative, display=True):
        pReturns = utils.getPeriodLogReturns(derivative.returns)
        mod_data = sm.tsa.MarkovAutoregression(
            pReturns["period"].values,
            k_regimes=2,
            order=1,
            trend="nc",
            switching_variance=True,
        )
        res_data = mod_data.fit()
        if display:
            displayResult(res_data.summary())
        return res_data
