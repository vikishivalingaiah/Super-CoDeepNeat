
class HistoricalMarker:
    def __init__(self):
        self.blueprint_marks = 0
        self.module_marks = 0
        self.blueprint_count = 0


    def mark_blueprint_genes(self):
        mark = self.blueprint_marks
        self.blueprint_marks = self.blueprint_marks + 1
        return mark

    def mark_blueprint(self):
        mark = self.blueprint_count
        self.blueprint_count = self.blueprint_count + 1
        return mark


    def mark_module(self):
        mark = self.module_marks
        self.module_marks = self.module_marks + 1
        return mark

