import pandas as pd
import numpy as np
from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
import quantutils.core.timeseries as tsUtils
from IPython.display import display as displayResult
import tradeframework.operations.plot as plotter
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import quantutils.core.statistics as stats


class AnalysisPlot(InsightGenerator):
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

        if display:
            plotter.tsplot(series, lags=self.opts["lags"])
        return series


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


class StationarityTest(InsightGenerator):
    """
    Test for Stationarity

    Where stationary data has:
    a) Constant mean
    b) Constant variance
    c) Constant covariance (No autocorrelation)
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("series", "returns")

    def getInsight(self, derivative, display=True):

        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        del self.opts["series"]
        result = stats.adf_test(series, self.opts)
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
            print("Number of lags used: {}".format(result[2]))
            print("Number of observations: {}".format(result[3]))
            print("T-Scores:")
            print("         -> 10% : {}".format(result[4]["10%"]))
            print("         -> 5%  : {}".format(result[4]["5%"]))
            print("         -> 1%  : {}".format(result[4]["1%"]))
            print()
            # print("Raw Output:")
            # print(result)
            # print()
            if result[1] > 0.1:
                print(
                    "H0 can be rejected with {}% confidence".format(
                        (1 - result[1]) * 100
                    )
                )
                print("Conclusion: Data is non-stationary")
            else:
                print(
                    "H0 can be rejected with {}% confidence".format(
                        (1 - result[1]) * 100
                    )
                )
                print("Conclusion: Data is stationary")
        return result


class WhiteNoiseTest(InsightGenerator):
    """
    Test for AutoCorrelations

    Utilises the Ljung-Box test
    https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("level", 0.95)
        self.opts.setdefault("series", "returns")
        self.opts.setdefault("sm_opts", {"lags": [20], "boxpierce": False})

    def getInsight(self, derivative, display=True):

        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        result = sm.stats.diagnostic.acorr_ljungbox(
            series, **self.opts["sm_opts"], return_df=False
        )
        if display:
            print()
            print("=============================================")
            print("Ljung-Box Test for AutoCorrelations")
            print("=============================================")
            print()
            print("H0 = No Serial AutoCorrelation in the data")
            print("H1 = AutoCorrelation exists")
            print()
            print("Critical value: {}".format(result[0][0]))
            print(f"Probability of White Noise: {result[1][0] * 100}%")
            print()
            print(f"H0 can be rejected with {1-result[1][0]}% confidence")
            if result[1][0] < (1 - self.opts["level"]):
                print("Conclusion: Data has autocorrelations")
            else:
                print("Conclusion: Data is white noise")

        return result


class NormalityTest(InsightGenerator):
    """
    Test for Normal Distribution of series

    Utilises the jaque-bera test for goodness of fit to Normal Distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("level", 0.95)
        self.opts.setdefault("series", "returns")
        self.opts.setdefault("sm_opts", {})

    def getInsight(self, derivative, display=True):

        if not isinstance(self.opts["series"], str):
            series = self.opts["series"]
        elif self.opts["series"] == "prices":
            series = np.log(derivative.values["Close"])
        elif self.opts["series"] == "returns":
            series = utils.getPeriodLogReturns(derivative.returns)["period"]

        result = sm.stats.stattools.jarque_bera(series, **self.opts["sm_opts"])
        if display:
            print()
            print("=============================================")
            print("Jaque-Bera Test for Normal Distribution")
            print("=============================================")
            print()
            print("H0 = Data is Normally distributed")
            print("H1 = Data may not be Normally distributed")
            print()
            print("Critical value: {}".format(result[0]))
            print(f"Probability of Normal Distribution: {result[1] * 100}%")
            print()
            print(f"H0 can be rejected with {1-result[1]}% confidence")
            if result[1] < (1 - self.opts["level"]):
                print("Conclusion: Data may not be Normally distributed")
            else:
                print("Conclusion: Data is Normally distributed")

        return result


class CorrelationMatrix(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("asset_list", None)

    def getInsight(self, derivative, display=True):
        result = utils.getPeriodLogReturns(derivative.returns).rename(
            columns={"period": derivative.getName()}
        )
        if self.opts["baseline"] is not None:
            result = result.join(
                utils.getPeriodLogReturns(self.opts["baseline"].returns).rename(
                    columns={"period": "Baseline"}
                )
            )
        if self.opts["asset_list"]:
            result = result.join(
                pd.concat(
                    [
                        utils.getPeriodLogReturns(
                            derivative.env.findAsset(assetName).returns
                        ).rename(columns={"period": assetName})
                        for assetName in self.opts["asset_list"]
                    ],
                    axis=1,
                )
            )
        else:
            result = result.join(
                pd.concat(
                    [
                        utils.getPeriodLogReturns(asset.returns).rename(
                            columns={"period": asset.getName()}
                        )
                        for asset in list(derivative.env.getAssetStore().store.values())
                    ],
                    axis=1,
                )
            )
        corr = result.corr()
        if display:
            displayResult(corr)
        return corr
