from tradeframework.api.insights import InsightGenerator
import quantutils.core.statistics as stats
import quantutils.dataset.pipeline as ppl
import quantutils.dataset.ml as mlUtils
import tradeframework.operations.utils as utils
import numpy as np
import tradeframework.operations.plot as plotter
import warnings
from IPython.display import display as displayResult

import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    r2_score,
)


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

        if display:
            print("=================================================")
            print("Model Parameters")
            print("=================================================")
            displayResult(result.params)
        return result


class PredictionPlot(InsightGenerator):
    """
    Create a summary plot of predictions vs actual.

    By default, this will plot derivative returns vs baseline returns.
    Can be used to provide alternative data, such as yhat vs y, or yhat vs residuals.
    If residuals, do not transform the predictions or residual to original data domain.
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if "baseline" not in opts and "actuals" not in opts:
            raise Exception("Missing parameter: baseline required")
        self.opts.setdefault("predictions", None)
        self.opts.setdefault("actuals", None)
        self.opts.setdefault("residuals", None)

    def getInsight(self, derivative, display=True):
        if self.opts["predictions"] is not None:
            x_series = self.opts["predictions"]
        else:
            x_series = utils.getPeriodLogReturns(derivative.returns)["period"]

        if self.opts["actuals"] is not None:
            y_series = self.opts["actuals"]
        else:
            y_series = utils.getPeriodLogReturns(self.opts["baseline"].returns)[
                "period"
            ]
        if self.opts["residuals"] is not None:
            residuals = self.opts["residuals"]
        else:
            residuals = y_series - x_series

        with plt.style.context("seaborn-darkgrid"):
            fig = plt.figure(figsize=(15, 10))

            layout = (2, 1)
            ax1 = plt.subplot2grid(layout, (0, 0), fig=fig)
            ax2 = plt.subplot2grid(layout, (1, 0), fig=fig)

            plotter.scatterPlot(
                x_series,
                y_series,
                title="Prediction Plot",
                x_axis="Predictions",
                y_axis="Actual values",
                ax=ax1,
                show=True,
            )
            plotter.scatterPlot(
                x_series,
                residuals,
                title="Residual Plot",
                x_axis="Predictions",
                y_axis="Residuals",
                ax=ax2,
                show=True,
            )

            plt.tight_layout()
            plt.close()

            tsFig = plotter.tsplot(
                residuals, lags=30, show=False, title="Residual Analysis"
            )

            if display:
                displayResult(fig)
                displayResult(tsFig)

        return fig


class PredictionMetrics(InsightGenerator):
    """
    Create metrics for predictions vs actual values

    MFE     Shows any +/- bias in the predictions. Ideally this value is 0
    MAE     To guage average prediction distance from actual
    MAPE    To guage average prediction distance from actual as a percentage (in practice not useful if close to 0)
    RSE     Residual Standard Error. The standard deviation of the errors (in units of the data)
            Useful for comparison of models on same dataset.
    MASE    To guage average prediction distance compared to average prediction distance of naive prediction (baseline).
            Useful for comparison of models across different datasets. If <1 then better than naive.
    R2      Proportion of data variance within the variance of the residuals.
    MSE     Only really useful as a loss function
    ME      Largest error
    MDA     Wins / Total
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("actual", None)
        self.opts.setdefault("predictions", None)
        self.opts.setdefault("ddof", 0)
        self.opts.setdefault("threshold", 0.5)

    def getInsight(self, derivative, display=True):
        if self.opts["predictions"] is not None:
            predictions = self.opts["predictions"]
        else:
            predictions = utils.getPeriodLogReturns(derivative.returns)["period"]

        if self.opts["actual"] is not None:
            actual = self.opts["actual"]
        else:
            actual = utils.getPeriodLogReturns(self.opts["baseline"].returns)["period"]

        mfe = stats.mean_forecast_err(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        mape = mean_absolute_percentage_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        me = max_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        my_mse = stats.mean_squared_err(actual, predictions)
        rse = stats.residual_standard_error(actual, predictions)
        my_mape = stats.mean_absolute_percentage_error(actual, predictions)
        my_mae = stats.mean_absolute_err(actual, predictions)
        mase = stats.mean_absolute_standard_error(actual, predictions)
        mda = stats.mean_directional_accuracy(actual, predictions)
        msa = stats.mean_sign_accuracy(actual, predictions)

        print()
        print("=================================================")
        print("Prediction Metrics")
        print("=================================================")
        print()
        print(f"Mean Forecast Error (MFE): {np.format_float_positional(mfe)}")
        print(f"Mean Absolute Error (MAE): {np.format_float_positional(mae)}")
        print(f"Max. Error: {np.format_float_positional(me)}")
        print(f"Residual Standard Error (RSE): {np.format_float_positional(rse)}")
        print(
            f"Mean Absolute Percentage Error (MAPE): {np.format_float_positional(mape)}"
        )
        print(
            f"Mean Absolute Standard Error (MASE): {np.format_float_positional(mase)}"
        )
        # print(f"Mean Squared Error (MSE): {np.format_float_positional(mse)}")
        print()
        print(f"R-Squared: {np.format_float_positional(r2)}")
        print(f"Mean Directional Accuracy (MDA): {np.format_float_positional(mda)}%")
        print(f"Mean Sign Accuracy (MSA): {np.format_float_positional(msa)}%")
        print()
        print("=================================================")
        print("Directional Metrics")
        print("=================================================")
        print()
        msa = mlUtils.evaluate(
            ppl.onehot(np.vstack(np.sign(actual))),
            ppl.onehot(np.vstack(np.sign(predictions))),
            threshold=self.opts["threshold"],
        )
        print(f"Accuracy: {msa:.2%}")
        print()
