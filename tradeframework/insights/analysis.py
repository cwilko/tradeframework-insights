import pandas as pd
import numpy as np
from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
import statsmodels.api as sm
import quantutils.core.statistics as stats


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
