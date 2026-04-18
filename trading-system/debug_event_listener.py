#!/usr/bin/env python3
"""
Subscribe to all Redis pub/sub channels and print structured events.

Usage:
    python debug_event_listener.py
"""

import asyncio
import json
import os

import redis.asyncio as redis


async def main() -> None:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    password = os.getenv("REDIS_PASSWORD")

    client = redis.Redis(host=host, port=port, db=db, password=password, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.psubscribe("*")
    print("Listening on all Redis channels...")
    try:
        async for message in pubsub.listen():
            if message.get("type") not in {"message", "pmessage"}:
                continue
            channel = message.get("channel")
            raw_payload = message.get("data")
            try:
                payload = json.loads(raw_payload)
            except Exception:
                payload = {"raw": raw_payload}
            print(
                json.dumps(
                    {
                        "event_type": payload.get("event_type"),
                        "channel": channel,
                        "correlation_id": payload.get("correlation_id"),
                        "payload": payload.get("payload", payload),
                    },
                    default=str,
                )
            )
    finally:
        await pubsub.close()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
