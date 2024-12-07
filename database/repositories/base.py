class BaseRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def _execute_transaction(self, operation):
        with self.session_factory() as session:
            try:
                result = operation(session)
                session.commit()
                return result
            except Exception as e:
                session.rollback()
                raise

    def add(self, entity):
        def operation(session):
            session.add(entity)
            return entity

        return self._execute_transaction(operation)

    def get_by_id(self, entity_class, entity_id):
        def operation(session):
            return session.query(entity_class).get(entity_id)

        return self._execute_transaction(operation)

    def update(self, entity):
        def operation(session):
            session.merge(entity)
            return entity

        return self._execute_transaction(operation)

    def delete(self, entity):
        def operation(session):
            session.delete(entity)

        return self._execute_transaction(operation)
