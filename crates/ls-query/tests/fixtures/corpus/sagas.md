# The Saga Pattern

The saga pattern coordinates a long-running business transaction across multiple
microservices by splitting it into a sequence of local transactions. Each local
transaction updates one service and publishes an event that triggers the next step
in the saga. There is no global lock and no two-phase commit.

# Compensating Transactions

When a step fails, the saga executes compensating transactions: previously
completed steps are undone in reverse order by explicit compensation logic. A
compensating transaction is not a rollback — it is a new action that semantically
reverses the effect, such as refunding a captured payment or releasing a held
seat reservation.

# Choreography and Orchestration

Sagas come in two coordination styles. In choreography, services react to each
other's events with no central coordinator; in orchestration, a dedicated saga
orchestrator tells each participant what to do and tracks progress in a state
machine. Orchestration is easier to reason about at scale, while choreography
minimizes coupling for small workflows.

# Compensation Failures

Compensation can itself fail, so compensating requests must be persisted and
retried until they succeed. Some actions cannot be compensated at all — sending
an email, for example — and should be deferred to the latest possible phase of
the saga so the likelihood of needing compensation is minimal.

