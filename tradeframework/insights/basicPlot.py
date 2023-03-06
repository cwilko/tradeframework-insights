from tradeframework.api.insights import InsightGenerator
import tradeframework.operations.plot as plotter


class BasicPlot(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        self.opts.setdefault("baseline", None)
        self.opts.setdefault("includePrimary", True)
        self.opts.setdefault("includeComponents", False)
        self.opts.setdefault("normalise", False)
        self.opts.setdefault("log", True)

    def getInsight(self, derivative, display=True):
        return plotter.plotReturns(derivative=derivative, **self.opts)
