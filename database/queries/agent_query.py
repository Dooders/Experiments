from database.models import Agent

class AgentQuery:
    def __init__(self, session):
        self.session = session
        self.query = session.query(Agent)

    def filter_by_type(self, agent_type: str):
        self.query = self.query.filter(Agent.agent_type == agent_type)
        return self

    def with_resources_above(self, threshold: float):
        self.query = self.query.filter(Agent.resource_level > threshold)
        return self

    def execute(self):
        return self.query.all()
