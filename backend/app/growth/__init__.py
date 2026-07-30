"""The Growth Engine — the sole writer to `app.memory2` (episodic/semantic
memory) and `app.relationship.state.RelationshipProfile`. Runs at the end
of a conversation, never synchronously mid-turn (Book Vol 3 Ch 11), and is
always invoked through its async entrypoint (`app.growth.engine.GrowthEngine`)."""
