#!/usr/bin/env python3
import json
import logging
import os
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import redis
import socks
from redis import ConnectionError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

# Assuming these are imported from other modules
from nebulous.processors.models import V1ProcessorHealthResponse


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
            # Removed StreamHandler to prevent health logs from appearing in main logs
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
    # print(f"[HEALTH DEBUG] === PROCESSING HEALTH CHECK MESSAGE ===")
    # print(f"[HEALTH DEBUG] Message ID: {message_id}")
    # print(f"[HEALTH DEBUG] Message data: {message_data}")
    # print(f"[HEALTH DEBUG] Health stream: {health_stream}")
    # print(f"[HEALTH DEBUG] Health group: {health_group}")

    logger.info(f"Processing health check message {message_id}: {message_data}")

    health_status = "ok"
    health_message: Optional[str] = "Health check processed successfully."
    details: Optional[Dict[str, Any]] = None
    return_stream: Optional[str] = None
    original_message_id: Optional[str] = None
    # user_id: Optional[str] = None

    try:
        if "data" in message_data:
            # print(f"[HEALTH DEBUG] Found 'data' field in message")
            data_str = message_data["data"]
            print(f"[HEALTH DEBUG] Raw data string: {data_str}")

            data = json.loads(data_str)
            # print(f"[HEALTH DEBUG] Parsed data: {json.dumps(data, indent=2)}")
            logger.info(f"Health check data: {data}")

            # Extract important fields from the forwarded message
            return_stream = data.get("return_stream")
            original_message_id = data.get("original_message_id")
            # user_id = data.get("user_id")
            # inner_kind = data.get("kind")
            inner_id = data.get("id")
            content = data.get("content")

            # print(f"[HEALTH DEBUG] Extracted fields:")
            # print(f"[HEALTH DEBUG]   return_stream: {return_stream}")
            # print(f"[HEALTH DEBUG]   original_message_id: {original_message_id}")
            # print(f"[HEALTH DEBUG]   user_id: {user_id}")
            # print(f"[HEALTH DEBUG]   inner_kind: {inner_kind}")
            # print(f"[HEALTH DEBUG]   inner_id: {inner_id}")
            # print(f"[HEALTH DEBUG]   content: {content}")

            # Update details with health check info
            details = {
                "checked_message_id": inner_id,
                "original_message_id": original_message_id,
                "health_stream": health_stream,
            }

            if content and isinstance(content, dict):
                details.update({"check_content": content})
                # print(f"[HEALTH DEBUG] Added check_content to details")
        else:
            # print(f"[HEALTH DEBUG] No 'data' field found in message_data")
            pass

    except (json.JSONDecodeError, KeyError) as e:
        # print(f"[HEALTH DEBUG] ERROR parsing message data: {e}")
        logger.warning(f"Could not parse health check message data: {e}")
        health_status = "error"
        health_message = f"Failed to parse health check message data: {e}"
        details = {"error": str(e)}

    # Construct the health response using V1ProcessorHealthResponse
    health_response = V1ProcessorHealthResponse(
        status=health_status, message=health_message, details=details
    )

    print(
        f"[HEALTH DEBUG] Constructed V1ProcessorHealthResponse: {health_response.model_dump_json()}"
    )
    logger.info(
        f"Health response for message {message_id}: {health_response.model_dump_json()}"
    )

    # Send the response to the return stream in the format expected by the system
    if return_stream:
        # print(
        #     f"[HEALTH DEBUG] Sending V1ProcessorHealthResponse to return_stream: {return_stream}"
        # )
        try:
            # Send the V1ProcessorHealthResponse directly (not wrapped in StreamResponseMessage)
            response_data = health_response.model_dump()

            # print(
            #     f"[HEALTH DEBUG] Sending V1ProcessorHealthResponse directly: {json.dumps(response_data, indent=2)}"
            # )

            # Send to return stream
            redis_conn.xadd(
                return_stream,
                {"data": json.dumps(response_data)},
                maxlen=1000,
                approximate=True,
            )
            # print(
            #     f"[HEALTH DEBUG] V1ProcessorHealthResponse sent successfully to {return_stream}"
            # )
            logger.info(
                f"Sent health response for {message_id} to stream: {return_stream}"
            )
        except Exception as e_resp_send:
            # print(f"[HEALTH DEBUG] ERROR sending response: {e_resp_send}")
            logger.error(
                f"Failed to send health response for {message_id} to stream {return_stream}: {e_resp_send}"
            )
    else:
        # print(f"[HEALTH DEBUG] No return_stream specified, not sending response")
        pass

    # Acknowledge the health check message
    # print(f"[HEALTH DEBUG] Acknowledging message {message_id}")
    try:
        redis_conn.xack(health_stream, health_group, message_id)
        # print(f"[HEALTH DEBUG] Message {message_id} acknowledged successfully")
        logger.info(f"Acknowledged health check message {message_id}")
    except Exception as e_ack:
        # print(f"[HEALTH DEBUG] ERROR acknowledging message: {e_ack}")
        logger.error(
            f"Failed to acknowledge health check message {message_id}: {e_ack}"
        )

    # print(f"[HEALTH DEBUG] === FINISHED PROCESSING HEALTH CHECK MESSAGE ===")


def main():
    """Main function for the health check consumer subprocess."""
    # print(f"[HEALTH DEBUG] === HEALTH WORKER STARTING ===")
    logger = setup_health_logging()

    # Get environment variables
    redis_url = os.environ.get("REDIS_URL")
    health_stream = os.environ.get("REDIS_HEALTH_STREAM")
    health_group = os.environ.get("REDIS_HEALTH_CONSUMER_GROUP")

    # print(f"[HEALTH DEBUG] Environment variables:")
    # print(f"[HEALTH DEBUG]   REDIS_URL: {redis_url}")
    # print(f"[HEALTH DEBUG]   REDIS_HEALTH_STREAM: {health_stream}")
    # print(f"[HEALTH DEBUG]   REDIS_HEALTH_CONSUMER_GROUP: {health_group}")

    if not all([redis_url, health_stream, health_group]):
        print(f"[HEALTH DEBUG] ERROR: Missing required environment variables")
        logger.error(
            "Missing required environment variables: REDIS_URL, REDIS_HEALTH_STREAM, REDIS_HEALTH_CONSUMER_GROUP"
        )
        sys.exit(1)

    # Type assertions after validation
    assert isinstance(redis_url, str)
    assert isinstance(health_stream, str)
    assert isinstance(health_group, str)

    print(
        f"[HEALTH DEBUG] Starting health consumer for stream: {health_stream}, group: {health_group}"
    )
    logger.info(
        f"Starting health consumer for stream: {health_stream}, group: {health_group}"
    )

    # Configure SOCKS proxy
    # print(f"[HEALTH DEBUG] Configuring SOCKS proxy...")
    socks.set_default_proxy(socks.SOCKS5, "localhost", 1055)
    socket.socket = socks.socksocket
    logger.info("Configured SOCKS5 proxy for socket connections via localhost:1055")

    health_redis_conn: Optional[redis.Redis] = None
    health_consumer_name = f"health-consumer-{os.getpid()}-{socket.gethostname()}"
    # print(f"[HEALTH DEBUG] Health consumer name: {health_consumer_name}")

    # print(f"[HEALTH DEBUG] === ENTERING MAIN LOOP ===")
    while True:
        try:
            if health_redis_conn is None:
                # print(f"[HEALTH DEBUG] Connecting to Redis for health stream...")
                logger.info("Connecting to Redis for health stream...")

                health_redis_conn = redis.from_url(redis_url, decode_responses=True)
                health_redis_conn.ping()
                # print(f"[HEALTH DEBUG] Connected to Redis successfully")
                logger.info("Connected to Redis for health stream.")

                # Create health consumer group if it doesn't exist
                # print(f"[HEALTH DEBUG] Creating/checking consumer group...")
                logger.info("Creating/checking consumer group...")
                try:
                    health_redis_conn.xgroup_create(
                        health_stream, health_group, id="0", mkstream=True
                    )
                    # print(
                    #     f"[HEALTH DEBUG] Created consumer group {health_group} for stream {health_stream}"
                    # )
                    logger.info(
                        f"Created consumer group {health_group} for stream {health_stream}"
                    )
                except ResponseError as e_group:
                    if "BUSYGROUP" in str(e_group):
                        # print(
                        #     f"[HEALTH DEBUG] Consumer group {health_group} already exists"
                        # )
                        logger.info(f"Consumer group {health_group} already exists.")
                    else:
                        # print(
                        #     f"[HEALTH DEBUG] ERROR creating health consumer group: {e_group}"
                        # )
                        logger.error(f"Error creating health consumer group: {e_group}")
                        time.sleep(5)
                        health_redis_conn = None
                        continue
                except Exception as e_group_other:
                    # print(
                    #     f"[HEALTH DEBUG] UNEXPECTED ERROR creating health consumer group: {e_group_other}"
                    # )
                    logger.error(
                        f"Unexpected error creating health consumer group: {e_group_other}"
                    )
                    time.sleep(5)
                    health_redis_conn = None
                    continue

            # Read from health stream
            assert health_redis_conn is not None

            # print(f"[HEALTH DEBUG] Reading from health stream {health_stream}...")
            logger.info(f"Reading from health stream {health_stream}...")
            health_streams_arg: Dict[str, object] = {health_stream: ">"}
            raw_messages = health_redis_conn.xreadgroup(
                health_group,
                health_consumer_name,
                health_streams_arg,  # type: ignore[arg-type]
                count=1,
                block=5000,  # Block for 5 seconds
            )

            # print(f"[HEALTH DEBUG] xreadgroup returned: {raw_messages}")
            # print(f"[HEALTH DEBUG] Messages type: {type(raw_messages)}")

            if raw_messages:
                # print(f"[HEALTH DEBUG] Found messages to process")
                logger.info("Found messages to process")
                # Cast to expected type for decode_responses=True
                messages = cast(
                    List[Tuple[str, List[Tuple[str, Dict[str, str]]]]], raw_messages
                )
                for stream_name, stream_messages in messages:
                    # print(
                    #     f"[HEALTH DEBUG] Processing stream: {stream_name} with {len(stream_messages)} message(s)"
                    # )
                    logger.info(
                        f"Processing stream: {stream_name} with {len(stream_messages)} message(s)"
                    )
                    for message_id, message_data in stream_messages:
                        # print(f"[HEALTH DEBUG] Processing message {message_id}")
                        logger.info(f"Processing message {message_id}")
                        process_health_check_message(
                            message_id,
                            message_data,
                            health_redis_conn,
                            logger,
                            health_stream,
                            health_group,
                        )
            else:
                # print(f"[HEALTH DEBUG] No messages received (timeout)")
                logger.info("No messages received (timeout)")

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
