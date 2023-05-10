class Layer(object):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.explored = False
        self.waiting = False
        self.completed = False

        # Updated through init of simulator
        self.dependencies = []  # all sources to this layer
        self.next = []  # all destinations from this layer
        self.size = 0
        self.possible_end_time = 0

        # Updated after partitions
        self.device_id = None
        self.arrival_time_pool = []
        self.end_time = 0
        self.pr_max = None
        self.pr_min = None
        self.fixed = None
