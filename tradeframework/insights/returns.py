from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
import quantutils.core.timeseries as tsUtils
import quantutils.core.statistics as stats
from IPython.display import display as displayResult
import tradeframework.operations.plot as plotter
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm


class RollingReturns(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("MA", {})
        self.opts["MA"].setdefault("window", 14)
        self.opts["MA"].setdefault("offset", int(self.opts["MA"]["window"] / 2))

        self.opts.setdefault("Std", {})
        self.opts["Std"].setdefault("window", 14)
        self.opts["Std"].setdefault("offset", int(self.opts["Std"]["window"] / 2))

        self.opts.setdefault("MACF", {})
        self.opts["MACF"].setdefault("lag", 1)
        self.opts["MACF"].setdefault("window", 14)
        self.opts["MACF"].setdefault("offset", int(self.opts["MACF"]["window"] / 2))

    def getInsight(self, derivative, display=True):
        opts = self.opts
        returns = utils.getPeriodReturns(derivative.returns)
        returns["Std"] = tsUtils.MStd(returns["period"].values, window=opts["Std"]["window"], offset=opts["Std"]["offset"])
        returns["Var"] = tsUtils.MVar(returns["period"].values, window=opts["Std"]["window"], offset=opts["Std"]["offset"])
        returns["AutoCorr"] = tsUtils.autocorr(returns["period"])
        returns["MACF"] = tsUtils.MACF(returns["period"].values, lag=opts["MACF"]["lag"], window=opts["MACF"]["window"], offset=opts["MACF"]["offset"])

        if display:
            displayResult(returns)
        return returns


class ReturnsPlot(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

    def getInsight(self, derivative, display=True):
        pReturns = utils.getPeriodReturns(derivative.returns)
        pReturns["axis"] = [0] * len(pReturns)
        pReturns["MA"] = tsUtils.MA(pReturns["period"].values, 20)
        if display:
            feeds = []
            feeds.append({"data": pReturns["period"], "opts": {"label": derivative.getName(), "color": "red", "marker": "o", "linestyle": "None"}})
            feeds.append({"data": pReturns["axis"], "opts": {"label": "_nolegend_"}})
            feeds.append({"data": pReturns["MA"], "opts": {"label": "MA = 20"}})
            plotter.basicPlot(title="Returns Plot", feeds=feeds)

        return pReturns
