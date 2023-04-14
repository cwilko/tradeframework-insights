import numpy as np
import pandas as pd

import quantutils.core.statistics as stats
from quantutils.core.plot import OHLCChart

import statsmodels.api as sm
from scipy import stats as scipy_stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import tradeframework.operations.utils as utils

import warnings
import pyfolio
from IPython.display import display

# Plot Candlestick chart for an asset


def plotAsset(asset, options=None):
    chart = OHLCChart(options)
    chart.addSeries(asset.getName(), asset.values)
    display(chart.getChart())
    return chart


def plotAssetByName(derivative, assetName, options=None):
    return plotAsset(derivative.findAsset(assetName), options)


def plotWeightedUnderlying(derivative, underlyingName, options=None):
    underlying = derivative.env.getAssetStore().getAsset(underlyingName)
    asset = underlying.values.mul(
        derivative.weights[underlyingName]["bar"].values, axis=0
    )
    asset.replace(0, np.nan, inplace=True)
    chart = OHLCChart(options)
    chart.addSeries(underlyingName, asset)
    display(chart.getChart())
    return chart


# Helper method to pretty print derviative performance


def plotReturns(
    derivative,
    baseline=None,
    log=True,
    includeComponents=False,
    includePrimary=True,
    normalise=False,
    custom=[],
):
    assets = []

    # Add component asset returns
    if includeComponents:
        assets = derivative.assets[:]

    # Add primary returns
    if includePrimary:
        assets.append(derivative)

    # Add baseline
    if baseline is not None:
        assets.append(baseline)

    for userdata in custom:
        assets.append(userdata)

    if log:

        def pnl(x):
            return np.cumsum(
                np.log(
                    (
                        utils.getPeriodReturns(
                            x.returns[np.prod(derivative.values, axis=1) != 1]
                        )
                        + 1
                    )
                    .resample("B")
                    .agg("prod")
                )
            )

    else:

        def pnl(x):
            return np.cumprod(
                (
                    utils.getPeriodReturns(
                        x.returns[np.prod(derivative.values, axis=1) != 1]
                    )
                    + 1
                )
                .resample("B")
                .agg("prod")
            )

    # quarters = MonthLocator([1, 3, 6, 9])
    # allmonths = MonthLocator()
    # mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    # alldays = DayLocator()              # minor ticks on the days
    # weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    auto_locator = AutoDateLocator()
    auto_formatter = AutoDateFormatter(auto_locator)

    matplotlib.rcParams["figure.figsize"] = (12.0, 6.0)
    plt.ion()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(auto_locator)
    # ax.xaxis.set_minor_locator(allmonths)
    ax.xaxis.set_major_formatter(auto_formatter)

    for asset in assets:
        data = pnl(asset)
        if normalise:
            data = data / data[-1]
        ax.plot(data, label=asset.name)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.title("Derivative performance")
    plt.legend(loc="best")
    # fig.canvas.draw()
    return fig


def basicPlot(title="Basic Plot", feeds=[]):
    auto_locator = AutoDateLocator()
    auto_formatter = AutoDateFormatter(auto_locator)

    matplotlib.rcParams["figure.figsize"] = (12.0, 6.0)
    plt.ion()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(auto_locator)
    # ax.xaxis.set_minor_locator(allmonths)
    ax.xaxis.set_major_formatter(auto_formatter)

    for feed in feeds:
        feed.setdefault("opts", {})
        ax.plot(feed["data"], **feed["opts"])

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.title(title)
    plt.legend(loc="best")
    # fig.canvas.draw()
    display(fig)
    return ax


def outlinePlot(title="Outline Plot"):
    matplotlib.rcParams["figure.figsize"] = (12.0, 6.0)
    plt.ion()
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    ax.autoscale_view()
    # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(title)
    plt.legend(loc="best")
    fig.canvas.draw()
    display(ax)
    return ax


def displaySummary(
    derivative,
    tInfo,
    baseline=None,
    log=False,
    includeComponents=True,
    includePrimary=True,
    full=True,
):
    print("Derivative name : " + derivative.name)
    print("Number of assets : " + str(len(derivative.assets)))
    if baseline is not None:
        print("Baseline name : " + baseline.name)

    # Summary plot
    plotReturns(derivative, baseline, log, includeComponents, includePrimary)

    # Show sample of trades
    pd.set_option("display.max_rows", 10)
    display(tInfo)

    if baseline is not None:
        # Show local statistics
        stats.merton(
            derivative.returns["Open"][derivative.returns["Open"] != 0],
            baseline.returns["Open"][baseline.returns["Open"] != 0],
            display=True,
        )

    if full:
        # Show generic statistics
        warnings.filterwarnings("ignore")
        pyfolio.create_returns_tear_sheet(utils.getPeriodReturns(derivative.returns))


def tsplot(
    y, lags=None, figsize=(15, 10), style="seaborn-darkgrid", title=None, show=False
):
    """
    For a summary of any time series data.
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (4, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
        hs_ax = plt.subplot2grid(layout, (1, 0), colspan=2, fig=fig)
        acf_ax = plt.subplot2grid(layout, (2, 0), fig=fig)
        pacf_ax = plt.subplot2grid(layout, (2, 1), fig=fig)
        qq_ax = plt.subplot2grid(layout, (3, 0), fig=fig)
        pp_ax = plt.subplot2grid(layout, (3, 1), fig=fig)

        y.plot(ax=ts_ax)
        description = "Time Series Analysis Plot"
        if title:
            description = f"{description}: {title}"
        ts_ax.set_title(description)
        plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line="s", ax=qq_ax)
        qq_ax.set_title("QQ Plot")
        scipy_stats.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        # scatterPlot(range(len(y)), y, ax=r_ax)
        histogram(y, ax=hs_ax, show=True)

        plt.tight_layout()
        if not show:
            plt.close()
    return fig


def scatterPlot(
    x,
    y,
    bestFit=True,
    figsize=(10, 6),
    style="seaborn-darkgrid",
    title="Scatter Plot",
    x_axis="x",
    y_axis="y",
    ax=None,
    show=False,
):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        if ax is None:
            ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, "o", label="data")

        if bestFit:
            a, b = np.polyfit(x, y, 1)
            ax.plot(x, a * x + b)
        ax.autoscale_view()
        # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment="right")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(title)
        # ax.legend(loc="best")
        if not show:
            plt.close()
        return ax.get_figure()


def histogram(
    x,
    figsize=(10, 6),
    style="seaborn-darkgrid",
    title="Histogram",
    x_axis="x",
    ax=None,
    show=False,
):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    with plt.style.context(style):
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        counts, bins = np.histogram(x, bins=100)
        # plt.stairs(counts, bins)
        ax.hist(bins[:-1], bins, weights=counts)

        ax.autoscale_view()
        # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment="right")
        ax.set_xlabel(x_axis)
        ax.set_title(title)
        # ax.legend(loc="best")
        if not show:
            plt.close()
        return ax.get_figure()
