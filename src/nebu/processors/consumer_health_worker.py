#!/usr/bin/env python3
import json
import logging
import os
import socket
import sys
import time
from typing import Dict, List, Optional, Tuple, cast

import redis
import socks
from redis import ConnectionError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError


def setup_health_logging():
    """Set up logging for the health check worker to write to a dedicated file."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with timestamp
    log_file = os.path.join(log_dir, f"health_consumer_{os.getpid()}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(
                sys.stdout
            ),  # Also log to stdout for subprocess monitoring
        ],
    )

    logger = logging.getLogger("HealthConsumer")
    logger.info(f"Health check worker started. Logging to: {log_file}")
    return logger


def process_health_check_message(
    message_id: str,
    message_data: Dict[str, str],
    redis_conn: redis.Redis,
    logger: logging.Logger,
    health_stream: str,
    health_group: str,
) -> None:
    """Processes a single health check message."""
    logger.info(f"Processing health check message {message_id}: {message_data}")

    # Parse the message if it contains JSON data
    try:
        if "data" in message_data:
            data = json.loads(message_data["data"])
            logger.info(f"Health check data: {data}")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not parse health check message data: {e}")

    # You could add more logic here, e.g., update an internal health status,
    # send a response, perform actual health checks, etc.

    # Acknowledge the health check message
    try:
        redis_conn.xack(health_stream, health_group, message_id)
        logger.info(f"Acknowledged health check message {message_id}")
    except Exception as e_ack:
        logger.error(
            f"Failed to acknowledge health check message {message_id}: {e_ack}"
        )


def main():
    """Main function for the health check consumer subprocess."""
    logger = setup_health_logging()

    # Get environment variables
    redis_url = os.environ.get("REDIS_URL")
    health_stream = os.environ.get("REDIS_HEALTH_STREAM")
    health_group = os.environ.get("REDIS_HEALTH_CONSUMER_GROUP")

    if not all([redis_url, health_stream, health_group]):
        logger.error(
            "Missing required environment variables: REDIS_URL, REDIS_HEALTH_STREAM, REDIS_HEALTH_CONSUMER_GROUP"
        )
        sys.exit(1)

    # Type assertions after validation
    assert isinstance(redis_url, str)
    assert isinstance(health_stream, str)
    assert isinstance(health_group, str)

    logger.info(
        f"Starting health consumer for stream: {health_stream}, group: {health_group}"
    )

    # Configure SOCKS proxy
    socks.set_default_proxy(socks.SOCKS5, "localhost", 1055)
    socket.socket = socks.socksocket
    logger.info("Configured SOCKS5 proxy for socket connections via localhost:1055")

    health_redis_conn: Optional[redis.Redis] = None
    health_consumer_name = f"health-consumer-{os.getpid()}-{socket.gethostname()}"

    while True:
        try:
            if health_redis_conn is None:
                logger.info("Connecting to Redis for health stream...")
                health_redis_conn = redis.from_url(redis_url, decode_responses=True)
                health_redis_conn.ping()
                logger.info("Connected to Redis for health stream.")

                # Create health consumer group if it doesn't exist
                try:
                    health_redis_conn.xgroup_create(
                        health_stream, health_group, id="0", mkstream=True
                    )
                    logger.info(
                        f"Created consumer group {health_group} for stream {health_stream}"
                    )
                except ResponseError as e_group:
                    if "BUSYGROUP" in str(e_group):
                        logger.info(f"Consumer group {health_group} already exists.")
                    else:
                        logger.error(f"Error creating health consumer group: {e_group}")
                        time.sleep(5)
                        health_redis_conn = None
                        continue
                except Exception as e_group_other:
                    logger.error(
                        f"Unexpected error creating health consumer group: {e_group_other}"
                    )
                    time.sleep(5)
                    health_redis_conn = None
                    continue

            # Read from health stream
            assert health_redis_conn is not None

            health_streams_arg: Dict[str, object] = {health_stream: ">"}
            raw_messages = health_redis_conn.xreadgroup(
                health_group,
                health_consumer_name,
                health_streams_arg,  # type: ignore[arg-type]
                count=1,
                block=5000,  # Block for 5 seconds
            )

            if raw_messages:
                # Cast to expected type for decode_responses=True
                messages = cast(
                    List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], raw_messages
                )
                for _stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        process_health_check_message(
                            message_id,
                            message_data,
                            health_redis_conn,
                            logger,
                            health_stream,
                            health_group,
                        )

        except (ConnectionError, RedisTimeoutError, TimeoutError) as e_conn:
            logger.error(f"Redis connection error: {e_conn}. Reconnecting in 5s...")
            if health_redis_conn:
                try:
                    health_redis_conn.close()
                except Exception:
                    pass
            health_redis_conn = None
            time.sleep(5)

        except ResponseError as e_resp:
            logger.error(f"Redis response error: {e_resp}")
            if "NOGROUP" in str(e_resp):
                logger.warning(
                    "Health consumer group disappeared. Attempting to recreate..."
                )
                if health_redis_conn:
                    try:
                        health_redis_conn.close()
                    except Exception:
                        pass
                health_redis_conn = None
            elif "UNBLOCKED" in str(e_resp):
                logger.info(
                    "XREADGROUP unblocked, connection might have been closed. Reconnecting."
                )
                if health_redis_conn:
                    try:
                        health_redis_conn.close()
                    except Exception:
                        pass
                health_redis_conn = None
                time.sleep(1)
            else:
                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down health consumer...")
            break

        except Exception as e:
            logger.error(f"Unexpected error in health check consumer: {e}")
            logger.exception("Traceback:")
            time.sleep(5)

    # Cleanup
    if health_redis_conn:
        try:
            health_redis_conn.close()
        except Exception:
            pass

    logger.info("Health check consumer shutdown complete.")


if __name__ == "__main__":
    main()
