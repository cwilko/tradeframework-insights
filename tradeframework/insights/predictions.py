from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.trader as trader
from IPython.display import display as displayResult


class Predictions(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        #self.opts.setdefault("prices", 1)
        self.opts.setdefault("underlying", False)
        self.opts.setdefault("capital", 1)
        self.opts.setdefault("target", None)
        self.opts.setdefault("filter", [])

    def getInsight(self, derivative, display=True):
        result = trader.predictSignals(derivative=derivative, **self.opts)
        if display:
            for i in range(len(result)):
                print()
                print("====================")
                print("Prediction " + str(i) + ":")
                trader.printSignals(result[i])
        return result
