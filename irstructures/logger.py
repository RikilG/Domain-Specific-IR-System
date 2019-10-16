class Logger:
    def __init__(self, level=None):
        self.level = level
    
    def log(self, message, level=5, end='\n'):
        if self.level is None and level<=5:
            print("Lvl:", level, "msg:", message, end=end)
        elif level<=self.level:
            print("Log:", message, end=end)