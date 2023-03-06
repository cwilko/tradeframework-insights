import pandas as pd
from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
import quantutils.core.timeseries as tsUtils
from IPython.display import display as displayResult
import tradeframework.operations.plot as plotter
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import quantutils.core.statistics as stats


class AutoCorrelationPlot(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("lags", 50)

    def getInsight(self, derivative, display=True):
        pReturns = utils.getPeriodReturns(derivative.returns)
        pReturns["ACF"] = tsUtils.autocorr(pReturns["period"])
        if display:
            ax = plotter.outlinePlot(title="AutoCorrelation Plot")
            plot_acf(x=pReturns["ACF"], ax=ax, **self.opts)
        return pReturns

# Moving AutoCorrelation Plot


class MACFPlot(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("lag", 1)
        self.opts.setdefault("window", 14)
        self.opts.setdefault("offset", int(self.opts["window"] / 2))

    def getInsight(self, derivative, display=True):
        pReturns = utils.getPeriodReturns(derivative.returns)
        pReturns["MACF"] = tsUtils.MACF(pReturns["period"].values, **self.opts)
        if display:
            feeds = []
            feeds.append({"data": pReturns["MACF"], "opts": {"label": derivative.getName(), "color": "red", "marker": "o", "linestyle": "None"}})
            plotter.basicPlot(title="Moving AutoCorrelation Plot", feeds=feeds)

        return pReturns

# Regime identification


class MarkovRegimeFit(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("k_regimes", 2)
        self.opts.setdefault("order", 1)
        self.opts.setdefault("trend", "nc")
        self.opts.setdefault("switching_variance", True)

    def getInsight(self, derivative, display=True):
        pReturns = utils.getPeriodReturns(derivative.returns)
        mod_data = sm.tsa.MarkovAutoregression(pReturns["period"].values, k_regimes=2, order=1, trend='nc', switching_variance=True)
        res_data = mod_data.fit()
        if display:
            displayResult(res_data.summary())
        return res_data

# For Stationarity


class ADFTest(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("alt_series", None)
        self.opts.setdefault("test_returns", False)

    def getInsight(self, derivative, display=True):
        series = derivative.values["Close"]
        if self.opts["test_returns"]:
            series = utils.getPeriodReturns(derivative.returns)["period"]
        if self.opts["alt_series"] is not None:
            series = self.opts["alt_series"]
        result = stats.adf_test(series)
        if display:
            print()
            print("=============================================")
            print("Augmented Dicker-Fuller Test for Stationarity")
            print("=============================================")
            print()
            print("H0 = Non-Stationary data")
            print("H1 = Stationary data")
            print()
            print("Critical value: {}".format(result[0]))
            print("Probability of Non-Stationarity: {}%".format(result[1] * 100))
            print("Number of observations: {}".format(result[3]))
            print("T-Scores:")
            print("         -> 10% : {}".format(result[4]["10%"]))
            print("         -> 5%  : {}".format(result[4]["5%"]))
            print("         -> 1%  : {}".format(result[4]["1%"]))
            print()
            if (result[1] > .1):
                print("H0 can be rejected with {}% confidence".format((1 - result[1]) * 100))
                print("Conclusion: Data is non-stationary")
            else:
                print("H0 can be rejected with {}% confidence".format((1 - result[1]) * 100))
                print("Conclusion: Data is stationary")
        return result


class CorrelationMatrix(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("asset_list", None)

    def getInsight(self, derivative, display=True):
        result = utils.getPeriodReturns(derivative.returns).rename(columns={"period": derivative.getName()})
        if self.opts["baseline"] is not None:
            result = result.join(utils.getPeriodReturns(self.opts["baseline"].returns).rename(columns={"period": "Baseline"}))
        if self.opts["asset_list"]:
            result = result.join(pd.concat([utils.getPeriodReturns(derivative.env.findAsset(assetName).returns).rename(columns={"period": assetName}) for assetName in self.opts["asset_list"]], axis=1))
        else:
            result = result.join(pd.concat([utils.getPeriodReturns(asset.returns).rename(columns={"period": asset.getName()}) for asset in list(derivative.env.getAssetStore().store.values())], axis=1))
        corr = result.corr()
        if display:
            displayResult(corr)
        return corr
