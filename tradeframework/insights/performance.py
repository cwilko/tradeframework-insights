from tradeframework.api.insights import InsightGenerator
import quantutils.core.statistics as stats
import tradeframework.operations.utils as utils
from IPython.display import display as displayResult
import warnings
import pyfolio


class PerfSummary(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

    def getInsight(self, derivative, display=True):
        returns = utils.getPeriodReturns(derivative.returns)["period"]
        if display:
            stats.statistics(ts=returns)
        return stats.getStats(returns)


class Merton(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if not opts["baseline"]:
            raise Exception("Missing parameter: baseline")

    def getInsight(self, derivative, display=True):
        returns = utils.getPeriodReturns(derivative.returns)["period"]
        baseline = utils.getPeriodReturns(self.opts["baseline"].returns)["period"]
        return stats.merton(model_ret=returns, baseline_ret=baseline, display=display)


class PyfolioSummary(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if not opts["baseline"]:
            raise Exception("Missing parameter: baseline")

    def getInsight(self, derivative, display=True):
        # Show generic statistics
        warnings.filterwarnings('ignore')
        pyfolio.create_returns_tear_sheet(utils.getPeriodReturns(derivative.returns)["period"])


class StatisticalTests(InsightGenerator):

    def __init__(self, name, opts):
        InsightGenerator.__init__(self, name, opts)

        if not opts["baseline"]:
            raise Exception("Missing parameter: baseline")

        self.opts.setdefault("level", 0.95)
        self.opts.setdefault("iterations", 1000)
        self.opts.setdefault("txCost", 0)

    def getInsight(self, derivative, display=True):
        sim_results = stats.bootstrap(ts=self.opts["baseline"].values, iterations=self.opts["iterations"], txCost=self.opts["txCost"])
        if display:
            stats.statistical_tests(utils.getPeriodReturns(derivative.returns)["period"], sim_results, self.opts["level"])