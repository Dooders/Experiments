class Resource:
    def __init__(self, resource_id, position, amount):
        self.resource_id = resource_id
        self.position = position  # (x, y) coordinates
        self.amount = amount

    def is_depleted(self):
        return self.amount <= 0

    def consume(self, consumption_amount):
        self.amount -= consumption_amount
        if self.amount < 0:
            self.amount = 0
