from tradeframework.api.insights import InsightGenerator
import quantutils.core.statistics as stats
import quantutils.core.timeseries as tsUtils
from IPython.display import display as displayResult


class RollingPrice(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("pricePoint", "Close")

        self.opts.setdefault("MA", {})
        self.opts["MA"].setdefault("window", 14)
        self.opts["MA"].setdefault("offset", int(self.opts["MA"]["window"] / 2))

        self.opts.setdefault("Std", {})
        self.opts["Std"].setdefault("window", 14)
        self.opts["Std"].setdefault("offset", int(self.opts["Std"]["window"] / 2))

        self.opts.setdefault("Stoch", {})
        self.opts["Stoch"].setdefault("window", 5)

    def getInsight(self, derivative, display=True):
        opts = self.opts
        prices = pd.DataFrame(derivative.values)
        pricePoints = prices[opts["pricePoint"]].values
        prices["MA"] = tsUtils.MA(pricePoints, window=opts["MA"]["window"], offset=opts["MA"]["offset"])
        prices["EMA"] = tsUtils.EMA(prices[opts["pricePoint"]], window=opts["MA"]["window"], offset=opts["MA"]["offset"])
        prices["Std"] = tsUtils.MStd(pricePoints, window=opts["Std"]["window"], offset=opts["Std"]["offset"])
        prices = tsUtils.stoch_osc(prices, periods=opts["Stoch"]["window"])

        if display:
            displayResult(prices)
        return prices
