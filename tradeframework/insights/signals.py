from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.trader as trader
from IPython.display import display as displayResult


class Signals(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("capital", 1)
        self.opts.setdefault("target", None)
        self.opts.setdefault("filter", [])

    def getInsight(self, derivative, display=True):
        result = trader.getCurrentSignal(derivative=derivative, **self.opts)
        if result and display:
            trader.printSignals(result)
        return result


class UnderlyingSignals(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("capital", 1)
        self.opts.setdefault("target", None)
        self.opts.setdefault("filter", [])

    def getInsight(self, derivative, display=True):
        result = []
        for asset in derivative.weightedAssets:
            signal = trader.getCurrentSignal(asset, **self.opts)
            if signal:
                result.append(signal)
        if display:
            trader.printSignals(result)
        return result
