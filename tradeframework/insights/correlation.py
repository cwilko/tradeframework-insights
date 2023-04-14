import pandas as pd
import numpy as np
from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.utils as utils
from IPython.display import display as displayResult
import seaborn
import matplotlib.pyplot as plt


class CorrelationMatrix(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("asset_list", None)
        self.opts.setdefault("alt_series", None)

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

        if self.opts["alt_series"]:
            result = self.opts["alt_series"]

        corr = result.corr()
        if display:
            displayResult(corr)
        return corr


class CorrelationMap(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("asset_list", None)
        self.opts.setdefault("alt_series", None)
        self.opts.setdefault("threshold", 0.8)

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

        if self.opts["alt_series"]:
            result = self.opts["alt_series"]

        corr = result.corr()
        with plt.style.context("seaborn-darkgrid"):
            _, ax = plt.subplots(figsize=(15, 10))
            seaborn.heatmap(
                corr,
                ax=ax,
                cmap="YlOrRd",
                mask=(np.abs(corr) <= self.opts["threshold"]),
                annot=True,
            )
            plt.close()

        if display:
            displayResult(ax.get_figure())

        return ax.get_figure()


class CorrelationPairPlot(InsightGenerator):
    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("asset_list", None)
        self.opts.setdefault("alt_series", None)
        self.opts.setdefault("threshold", 0.8)

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

        if self.opts["alt_series"]:
            result = self.opts["alt_series"]

        with plt.style.context("seaborn-darkgrid"):
            pairMap = seaborn.PairGrid(result)
            pairMap.map_diag(seaborn.histplot)
            pairMap.map_offdiag(seaborn.scatterplot)
            pairMap.figure.set_size_inches(12, 8)
            plt.close()

        if display:
            displayResult(pairMap.figure)

        return pairMap.figure
