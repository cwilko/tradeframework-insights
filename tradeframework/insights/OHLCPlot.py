from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.plot as plotter


class OHLCPlot(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("options", None)

    def getInsight(self, derivative, display=True):
        return plotter.plotAsset(asset=derivative, **self.opts)


class OHLCPlotByName(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if not opts["assetName"]:
            raise Exception("Missing parameter: assetName")

        self.opts.setdefault("options", None)

    def getInsight(self, derivative, display=True):
        return plotter.plotAssetByName(derivative=derivative, **self.opts)


class OHLCPlotWeightedUnderlying(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if not opts["underlyingName"]:
            raise Exception("Missing parameter: underlyingName")

        self.opts.setdefault("options", None)

    def getInsight(self, derivative, display=True):
        return plotter.plotWeightedUnderlying(derivative=derivative, **self.opts)
