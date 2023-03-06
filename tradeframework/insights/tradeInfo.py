from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.trader as trader
from IPython.display import display as displayResult


class TradeInfo(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("startCapital", 1)
        self.opts.setdefault("unitAllocations", True)
        self.opts.setdefault("summary", True)

    def getInsight(self, derivative, display=True):
        result = trader.getTradingInfo(derivative=derivative, **self.opts)
        if display:
            displayResult(result)
        return result


class UnderlyingAllocations(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

    def getInsight(self, derivative, display=True):
        result = trader.getUnderlyingAllocations(derivative=derivative, **self.opts)
        if display:
            displayResult(result)
        return result
