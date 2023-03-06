import importlib


class InsightGenerator:

    def __init__(self, name, opts):
        self.name = name
        self.opts = opts

    def getName(self):
        return self.name

    def getInsight(self, derivative, display=True):
        pass


class InsightManager:

    def __init__(self, derivative):
        self.derivative = derivative
        self.generators = []

    def createInsightGenerator(self, generatorClass, generatorName=None, generatorModule="tradeframework.insights", opts=None):
        if not opts:
            opts = {}
        if not generatorName:
            generatorName = generatorClass
        module = importlib.import_module(generatorModule)
        generatorInstance = getattr(module, generatorClass)
        generator = generatorInstance(generatorName, opts)
        return generator

    def addInsightGenerator(self, generator):
        self.generators.append(generator)
        return self

    def generateInsights(self, display=True):
        insights = {}
        [insights.update({generator.getName(): generator.getInsight(self.derivative, display=display)}) for generator in self.generators]
        return insights
