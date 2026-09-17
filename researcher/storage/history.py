from researcher.models import ResearchSession


async def save_session(pool, session: ResearchSession) -> int:
    return await pool.fetchval(
        """INSERT INTO research_sessions
           (question, answer, sources_used, sources_failed, duration_ms)
           VALUES ($1, $2, $3, $4, $5) RETURNING id""",
        session.question, session.answer,
        session.sources_used, session.sources_failed, session.duration_ms,
    )


async def list_sessions(pool, limit: int = 10) -> list[ResearchSession]:
    rows = await pool.fetch(
        "SELECT * FROM research_sessions ORDER BY created_at DESC LIMIT $1",
        limit,
    )
    return [ResearchSession(**dict(r)) for r in rows]
