from tradeframework.api.insights import InsightGenerator
import quantutils.core.statistics as stats
import quantutils.dataset.pipeline as ppl
import quantutils.dataset.ml as mlUtils
import tradeframework.operations.utils as utils
import numpy as np
import tradeframework.operations.plot as plotter
from IPython.display import display as displayResult

import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)


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
        rse = stats.residual_standard_error(actual, predictions)
        mase = stats.mean_absolute_standard_error(actual, predictions)
        mda = stats.mean_directional_accuracy(actual, predictions)
        msa = stats.mean_sign_accuracy(actual, predictions)

        print()
        print("=================================================")
        print("Regression Metrics")
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


class ConfusionMatrix(InsightGenerator):
    """
    Create a plot of a Confusion Matrix

    By default, this will plot for derivative returns vs baseline returns.
    Can be used to provide alternative data.
    """

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("actual", None)
        self.opts.setdefault("predictions", None)
        self.opts.setdefault("noHold", False)  # Eliminate 0 value entries
        self.opts.setdefault("normalize", None)
        self.opts.setdefault("returnsData", True)  # Convert to predictions?

    def getInsight(self, derivative, display=True):
        if self.opts["predictions"] is not None:
            predicted = self.opts["predictions"]
        else:
            predicted = utils.getPeriodLogReturns(derivative.returns)["period"]
            self.opts["returnsData"] = True

        if self.opts["actual"] is not None:
            actuals = self.opts["actual"]
        else:
            actuals = utils.getPeriodLogReturns(self.opts["baseline"].returns)["period"]
            self.opts["returnsData"] = True

        if self.opts["noHold"]:
            index = actuals[actuals != 0].index.intersection(
                predicted[predicted != 0].index
            )
            actuals = actuals.loc[index]
            predicted = predicted.loc[index]
            displayLabels = ["Buy", "Sell"]
            labels = [1, -1]
        else:
            displayLabels = ["Buy", "Hold", "Sell"]
            abels = [1, 0, -1]

        actual = np.sign(actuals)
        predictions = np.sign(predicted)

        # Flip the sign if the predictions are returns and the baseline went down.
        # If baseline went down, and our return went up, we must have predicted down correctly (True Negative)
        # If baseline went down, and our return went down, we must have predicted up incorrectly (False Negative)
        if self.opts["returnsData"]:
            predictions[actual == -1] = np.negative(predictions[actual == -1])

        cf = confusion_matrix(
            actual, predictions, labels=labels, normalize=self.opts["normalize"]
        )

        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=cf, display_labels=displayLabels
        )

        if display:
            wins = np.sum(actual == predictions)
            total = len(actual)
            meanBuy = np.exp(actuals[actuals > 0].mean()) - 1
            meanSell = np.exp(actuals[actuals < 0].mean()) - 1

            cf_norm = cf
            if self.opts["normalize"] is None:
                cf_norm = confusion_matrix(
                    actual, predictions, labels=labels, normalize="all"
                )

            # Print results
            print()
            print("=================================================")
            print("Classification Metrics")
            print("=================================================")
            print()
            print(f"Won : {wins}")
            print(f"Lost : {total - wins}")
            print(f"Total : {total}")
            print(f"Diff : {wins - (total - wins)}")
            print()
            """
            Note that Accuracy is the expected value - the Confusion Matrix multiplied by the identify matrix 
            CF * [1, 0]  = accuracy
                 [0, 1] 
            """
            print(f"Accuracy : {(wins / total):.2%}")
            """
            Information Co-efficient - https://www.investopedia.com/terms/i/information-coefficient.asp

            Note that IC/Edge is the expected value - the Confusion Matrix multiplied by the following matrix 
            CF * [1, -1]  = IC
                 [-1, 1] 
            """
            print(f"Information Coefficient (Edge): {(((wins / total) * 2) - 1):.2%}")
            """
            Expected value - the Confusion Matrix multiplied by mean return of the baseline
            """
            print(
                f"Expected Value (Annualised): {np.sum(cf_norm * np.array([[meanBuy, -(meanBuy)],[meanSell, -(meanSell)]])) * 252:.2%}"
            )
            print()
            print("Precision: Of all the predicted Buys/Sells, how many were correct?")
            print(
                f"Precision (Buy) : {precision_score(actual, predictions, pos_label=1):.2%}"
            )
            print(
                f"Precision (Sell): {precision_score(actual, predictions, pos_label=-1):.2%}"
            )
            print()
            print("Recall: Of all the actual Buys/Sells, how many were correct?")
            print(f"Recall (Buy): {recall_score(actual, predictions, pos_label=1):.2%}")
            print(
                f"Recall (Sell): {recall_score(actual, predictions, pos_label=-1):.2%}"
            )
            print()
            """
            Note that F1 is the expected value of the Confusion Matrix multiplied by some unknown matrix, which
            weights the positive outcome. The matrix can be determined from regression. 
            See https://towardsdatascience.com/is-f1-score-really-better-than-accuracy-5f87be75ae01
            CF * [high, high]  = F1
                 [low,  low] 
            """
            print("F1 Score: Harmonic mean of Precision and Recall for the Buys/Sells")
            print(f"F1 Score (Buy): {f1_score(actual, predictions, pos_label=1):.2%}")
            print(f"F1 Score (Sell): {f1_score(actual, predictions, pos_label=-1):.2%}")

            cm_display.plot()
            plt.close()
            displayResult(cm_display.figure_)

        return cf
