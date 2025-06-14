import fnmatch
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from nebulous.logging import logger


# For RcloneBucket with direct subprocess calls
def is_rclone_installed() -> bool:
    """Check if rclone is installed and available in the PATH."""
    try:
        result = subprocess.run(
            ["rclone", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


import logging  # For logging.DEBUG etc.


def rclone_copy(
    source_dir: str,
    destination: str,
    dry_run: bool = False,
    transfers: int = 4,
    extra_args: Optional[List[str]] = None,
    verbose: bool = True,
) -> bool:
    """
    Upload a directory to a remote bucket using `rclone copy`.

    Args:
        source_dir (str): Path to local directory to upload.
        destination (str): Remote destination, e.g., 's3:my-bucket/path'.
        dry_run (bool): If True, performs a dry run without uploading.
        transfers (int): Number of parallel transfers.
        extra_args (Optional[List[str]]): Additional rclone flags.
        verbose (bool): If True, prints command and output live.

    Returns:
        bool: True if upload succeeded, False otherwise.
    """
    command = [
        "rclone",
        "copy",
        source_dir,
        destination,
        f"--transfers={transfers}",
        # "--progress",
    ]

    if dry_run:
        command.append("--dry-run")
    if extra_args:
        command.extend(extra_args)

    if verbose:
        logger.info(f"Running command: {' '.join(command)}")

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        if not process.stdout:
            raise Exception("No output from rclone")

        for line in process.stdout:
            if verbose:
                logger.debug(line.strip())

        return process.wait() == 0

    except Exception as e:
        logger.error(f"Error during rclone copy: {e}")
        return False


def find_latest_checkpoint(training_dir: str) -> Optional[str]:
    """
    Finds the checkpoint directory with the highest step number in a Hugging Face
    training output directory.

    Args:
        training_dir (str): The path to the training output directory.

    Returns:
        Optional[str]: The path to the latest checkpoint directory, or None if
                       no checkpoint directories are found or the directory
                       doesn't exist.
    """
    latest_step = -1
    latest_checkpoint_dir = None

    if not os.path.isdir(training_dir):
        logger.error(f"Error: Directory not found: {training_dir}")
        return None

    for item in os.listdir(training_dir):
        item_path = os.path.join(training_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                step_str = item.split("-")[-1]
                if step_str.isdigit():
                    step = int(step_str)
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint_dir = item_path
            except (ValueError, IndexError):
                # Ignore items that don't match the expected pattern
                continue

    return latest_checkpoint_dir


def _parse_s3_path(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Standalone helper: Parses an S3 path (s3://bucket/prefix) into bucket and prefix."""
    parsed = urlparse(path)
    if parsed.scheme != "s3":
        return None, None
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


class StorageBucket(ABC):
    """Abstract base class for bucket operations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def sync(
        self,
        source: str,
        destination: str,
        delete: bool = False,
        dry_run: bool = False,
        excludes: Optional[List[str]] = None,
    ) -> None:
        """
        Synchronizes files between a source and a destination.
        """
        pass

    @abstractmethod
    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """
        Copies files or directories between a source and a destination.
        """
        pass

    @abstractmethod
    def check(self, path_uri: str) -> bool:
        """
        Checks if an object or prefix exists.
        """
        pass


class S3Bucket(StorageBucket):
    """Handles interactions with AWS S3."""

    def __init__(
        self,
        verbose: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """
        Initializes the S3 handler. Can use default credentials or provided temporary ones.

        Args:
            verbose (bool): If True, prints status messages. Defaults to True.
            aws_access_key_id (Optional[str]): Temporary AWS Access Key ID.
            aws_secret_access_key (Optional[str]): Temporary AWS Secret Access Key.
            aws_session_token (Optional[str]): Temporary AWS Session Token (required if keys are temporary).
            region (str): AWS region for S3 operations. Defaults to "us-east-1".
        """
        super().__init__(verbose=verbose)
        if aws_access_key_id and aws_secret_access_key:
            if self.verbose:
                logger.info(
                    "Initializing S3 client with provided temporary credentials."
                )
            self.client = boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,  # Pass session token if provided
            )
        else:
            if self.verbose:
                logger.info("Initializing S3 client with default credentials.")
            self.client = boto3.client("s3", region_name=region)

    def _parse_path(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Class method: Parses an S3 path (s3://bucket/prefix) into bucket and prefix."""
        # Reusing the standalone logic here for consistency
        return _parse_s3_path(path)

    def _list_objects(
        self, bucket: str, prefix: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Class method: Lists objects in an S3 prefix."""
        objects: Dict[str, Dict[str, Any]] = {}
        paginator = self.client.get_paginator("list_objects_v2")
        list_prefix = prefix or ""
        if self.verbose:
            logger.info(f"Listing objects in s3://{bucket}/{list_prefix}...")

        operation_parameters = {"Bucket": bucket}
        if list_prefix:
            operation_parameters["Prefix"] = list_prefix

        try:
            page_iterator = paginator.paginate(**operation_parameters)
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if obj["Key"].endswith("/") and obj["Size"] == 0:
                            continue
                        relative_key: Optional[str] = None
                        current_prefix = prefix or ""
                        if current_prefix and obj["Key"].startswith(current_prefix):
                            prefix_adjusted = current_prefix + (
                                "" if current_prefix.endswith("/") else "/"
                            )
                            if obj["Key"] == current_prefix.rstrip("/"):
                                relative_key = os.path.basename(obj["Key"])
                            elif obj["Key"].startswith(prefix_adjusted):
                                relative_key = obj["Key"][len(prefix_adjusted) :]
                            else:
                                potential_rel_key = obj["Key"][len(current_prefix) :]
                                relative_key = potential_rel_key.lstrip("/")
                        elif not current_prefix:
                            relative_key = obj["Key"]
                        if not relative_key:
                            continue
                        last_modified = obj["LastModified"]
                        if last_modified.tzinfo is None:
                            last_modified = last_modified.replace(tzinfo=timezone.utc)
                        objects[relative_key] = {
                            "path": f"s3://{bucket}/{obj['Key']}",
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "mtime": last_modified,
                            "type": "s3",
                        }
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchBucket":
                if self.verbose:
                    logger.error(f"Error: Bucket '{bucket}' not found.")
            elif e.response["Error"]["Code"] == "NoSuchKey" and prefix:
                if self.verbose:
                    logger.warning(
                        f"Prefix s3://{bucket}/{prefix} not found (treating as empty)."
                    )
            else:
                logger.error(f"Error listing S3 objects: {e}")
            if e.response["Error"]["Code"] == "NoSuchBucket":
                return {}
        except Exception as e:
            logger.error(f"An unexpected error occurred listing S3 objects: {e}")
            return {}
        if self.verbose:
            logger.info(f"Found {len(objects)} objects in S3.")
        return objects

    def _list_local(
        self, local_dir: str, excludes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Class method: Lists files in a local directory."""
        files: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(local_dir):
            if self.verbose:
                logger.warning(
                    f"Warning: Local path not found: {local_dir} (treating as empty)."
                )
            return files
        if os.path.isfile(local_dir):
            if self.verbose:
                logger.warning(
                    f"Warning: Source {local_dir} is a file, not a directory. Syncing single file."
                )
            try:
                file_name = os.path.basename(local_dir)
                files[file_name] = {
                    "path": local_dir,
                    "size": os.path.getsize(local_dir),
                    "mtime": datetime.fromtimestamp(
                        os.path.getmtime(local_dir), tz=timezone.utc
                    ),
                    "type": "local",
                }
            except OSError as e:
                logger.error(f"Error accessing source file {local_dir}: {e}")
            return files
        if self.verbose:
            logger.info(f"Scanning local directory: {local_dir}...")
        for root, dirs, file_list in os.walk(local_dir):
            # Exclude __pycache__ directories
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")

            # Apply custom excludes for directories
            if excludes:
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(fnmatch.fnmatch(d, pattern) for pattern in excludes)
                ]

            for file_name in file_list:
                # Exclude .pyc files
                if file_name.endswith(".pyc"):
                    continue

                # Apply custom excludes for files
                if excludes and any(
                    fnmatch.fnmatch(file_name, pattern) for pattern in excludes
                ):
                    continue

                # Also check full relative path for excludes
                # This allows patterns like 'subdir/*' or '*.log' to work across the tree
                potential_relative_path = os.path.relpath(
                    os.path.join(root, file_name), local_dir
                ).replace("\\", "/")
                if excludes and any(
                    fnmatch.fnmatch(potential_relative_path, pattern)
                    for pattern in excludes
                ):
                    continue

                local_path = os.path.join(root, file_name)
                try:
                    relative_path = os.path.relpath(local_path, local_dir).replace(
                        "\\", "/"
                    )
                    files[relative_path] = {
                        "path": local_path,
                        "size": os.path.getsize(local_path),
                        "mtime": datetime.fromtimestamp(
                            os.path.getmtime(local_path), tz=timezone.utc
                        ),
                        "type": "local",
                    }
                except OSError as e:
                    logger.warning(
                        f"Warning: Could not get metadata for {local_path}: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Warning: Unexpected error processing {local_path}: {e}"
                    )
        if self.verbose:
            logger.info(f"Found {len(files)} files locally.")
        return files

    def sync(
        self,
        source: str,
        destination: str,
        delete: bool = False,
        dry_run: bool = False,
        excludes: Optional[List[str]] = None,
    ) -> None:
        """
        Synchronizes files between a source and a destination (local or S3).
        Compares file sizes and modification times. Copies if missing, larger, or newer.
        Optionally deletes extraneous files from the destination.
        Args:
            source (str): The source path (local directory/file or s3://...).
            destination (str): The destination path (local directory or s3://...).
            delete (bool): If True, delete extraneous files from the destination.
            dry_run (bool): If True, print actions without performing them.
            excludes (Optional[List[str]]): List of patterns to exclude from sync.
        """
        mtime_tolerance = timedelta(seconds=2)
        src_bucket, src_prefix = self._parse_path(source)
        dest_bucket, dest_prefix = self._parse_path(destination)
        source_items: Dict[str, Dict[str, Any]] = {}
        dest_items: Dict[str, Dict[str, Any]] = {}
        sync_direction = ""
        is_single_file_sync = False

        if src_bucket is None and dest_bucket is not None:
            sync_direction = "upload"
            source_items = self._list_local(source, excludes=excludes)
            dest_items = self._list_objects(dest_bucket, dest_prefix)
            if not source_items and not os.path.exists(source):
                logger.warning(
                    f"Warning: Source path {source} not found, but proceeding as it might be an empty source sync."
                )  # Check needed? list_local handles it.
                # return # Let it proceed if source is just empty
            if os.path.isfile(source):
                is_single_file_sync = True
            # current_dest_prefix = dest_prefix or "" # Moved closer to usage

        elif src_bucket is not None and dest_bucket is None:
            sync_direction = "download"
            source_items = self._list_objects(src_bucket, src_prefix)
            if os.path.exists(destination) and not os.path.isdir(destination):
                logger.error(
                    f"Error: Local destination '{destination}' exists but is not a directory."
                )
                return
            dest_items = self._list_local(destination, excludes=excludes)
            if not dry_run:
                os.makedirs(destination, exist_ok=True)
            elif not os.path.isdir(destination) and self.verbose:
                logger.info(f"Dry run: Would create local directory {destination}")

        elif src_bucket is None and dest_bucket is None:
            logger.error(
                "Error: Both source and destination are local paths. Use standard file copy tools."
            )
            return
        elif src_bucket is not None and dest_bucket is not None:
            logger.error(
                "Error: S3 to S3 sync not implemented. Use AWS CLI or S3 Batch Operations."
            )
            return
        else:
            logger.error("Error: Invalid source or destination path combination.")
            return

        actions_to_perform: List[Dict[str, Any]] = []
        source_keys = set(source_items.keys())
        dest_keys = set(dest_items.keys())

        for rel_key in source_keys:
            src_item = source_items[rel_key]
            dest_item = dest_items.get(rel_key)
            reason = ""
            if dest_item is None:
                reason = "does not exist in destination"
            else:
                if src_item["size"] != dest_item["size"]:
                    reason = f"size differs (src: {src_item['size']}, dest: {dest_item['size']})"
                elif src_item["mtime"] > (dest_item["mtime"] + mtime_tolerance):
                    reason = f"is newer in source (src: {src_item['mtime']}, dest: {dest_item['mtime']})"
            if reason:
                action_type = "upload" if sync_direction == "upload" else "download"
                dest_full_path_or_key: Optional[str] = None
                if sync_direction == "upload":
                    # Define current_dest_prefix here, just before use
                    current_dest_prefix = dest_prefix or ""
                    final_dest_key = (
                        rel_key
                        if is_single_file_sync
                        else os.path.join(current_dest_prefix, rel_key).replace(
                            "\\", "/"
                        )
                    )
                    if not current_dest_prefix and final_dest_key.startswith("/"):
                        final_dest_key = final_dest_key.lstrip("/")
                    dest_full_path_or_key = f"s3://{dest_bucket}/{final_dest_key}"
                else:
                    dest_full_path_or_key = os.path.join(
                        destination, rel_key.replace("/", os.sep)
                    )
                actions_to_perform.append(
                    {
                        "action": action_type,
                        "relative_key": rel_key,
                        "source_path": src_item["path"],
                        "source_mtime": src_item.get("mtime"),
                        "dest_full_path_or_key": dest_full_path_or_key,
                        "dest_bucket": dest_bucket,
                        "dest_prefix": dest_prefix,
                        "s3_key_full_src": src_item.get("key")
                        if sync_direction == "download"
                        else None,
                        "source_bucket": src_bucket,
                        "reason": reason,
                    }
                )

        if delete:
            keys_to_delete = dest_keys - source_keys
            for rel_key in keys_to_delete:
                dest_item = dest_items[rel_key]
                action_type = (
                    "delete_s3" if sync_direction == "upload" else "delete_local"
                )
                actions_to_perform.append(
                    {
                        "action": action_type,
                        "relative_key": rel_key,
                        "path_to_delete": dest_item["path"],
                        "s3_key_full_dest": dest_item.get("key")
                        if sync_direction == "upload"
                        else None,
                        "dest_bucket": dest_bucket,
                        "reason": "does not exist in source",
                    }
                )

        uploads_done = downloads_done = deletions_done = 0
        s3_deletions_batch: List[Dict[str, str]] = []
        if not actions_to_perform:
            if self.verbose:
                logger.info("Source and destination are already synchronized.")
            # Optional: Add check if source exists if sync_direction == "upload" and not os.path.exists(source):
            return

        for action in actions_to_perform:
            reason = action["reason"]

            if action["action"] == "upload":
                local_path = action["source_path"]
                dest_full_path_or_key = action["dest_full_path_or_key"]
                if not isinstance(dest_full_path_or_key, str):
                    logger.error(f"ERROR: Invalid dest path: {dest_full_path_or_key}")
                    continue
                _, upload_key = self._parse_path(dest_full_path_or_key)
                target_bucket = action["dest_bucket"]
                if self.verbose:
                    logger.info(
                        f"Upload: {local_path} to {dest_full_path_or_key} ({reason})"
                    )
                if not dry_run:
                    if target_bucket and upload_key is not None:
                        try:
                            self.client.upload_file(
                                local_path, target_bucket, upload_key
                            )
                            uploads_done += 1
                        except ClientError as e:
                            logger.error(f"ERROR uploading {local_path}: {e}")
                        except Exception as e:
                            logger.error(f"ERROR uploading {local_path}: {e}")
                    else:
                        logger.error(
                            f"ERROR: Invalid S3 target: bucket={target_bucket}, key={upload_key}"
                        )
            elif action["action"] == "download":
                s3_key_full = action["s3_key_full_src"]
                local_path = action["dest_full_path_or_key"]
                source_bucket_dl = action["source_bucket"]
                if self.verbose:
                    logger.info(
                        f"Download: {action['source_path']} to {local_path} ({reason})"
                    )
                if not isinstance(local_path, str):
                    logger.error(f"ERROR: Invalid local dest path: {local_path}")
                    continue
                if not dry_run:
                    if source_bucket_dl and s3_key_full and local_path:
                        try:
                            local_file_dir = os.path.dirname(local_path)
                            os.makedirs(local_file_dir, exist_ok=True)
                            self.client.download_file(
                                source_bucket_dl, s3_key_full, local_path
                            )
                            downloads_done += 1
                        except ClientError as e:
                            logger.error(f"ERROR downloading {s3_key_full}: {e}")
                        except OSError as e:
                            logger.error(f"ERROR creating/writing {local_path}: {e}")
                        except Exception as e:
                            logger.error(f"ERROR downloading {s3_key_full}: {e}")
                    else:
                        logger.error(
                            f"ERROR: Invalid download params: bucket={source_bucket_dl}, key={s3_key_full}, local={local_path}"
                        )
            elif action["action"] == "delete_s3":
                s3_key_to_delete = action["s3_key_full_dest"]
                target_bucket_del = action["dest_bucket"]
                if target_bucket_del and s3_key_to_delete:
                    if self.verbose:
                        logger.info(f"Delete S3: {action['path_to_delete']} ({reason})")
                    if isinstance(s3_key_to_delete, str):
                        s3_deletions_batch.append({"Key": s3_key_to_delete})
                    else:
                        logger.error(
                            f"ERROR: Invalid S3 key for deletion: {s3_key_to_delete}"
                        )
                else:
                    logger.error(
                        f"ERROR: Invalid S3 target for deletion: bucket={target_bucket_del}, key={s3_key_to_delete}"
                    )
            elif action["action"] == "delete_local":
                local_path_to_delete = action["path_to_delete"]
                if self.verbose:
                    logger.info(f"Delete Local: {local_path_to_delete} ({reason})")
                if not dry_run:
                    try:
                        os.remove(local_path_to_delete)
                        deletions_done += 1
                    except OSError as e:
                        logger.error(
                            f"ERROR deleting local file {local_path_to_delete}: {e}"
                        )

        if s3_deletions_batch:
            target_bucket_del_batch = next(
                (
                    a["dest_bucket"]
                    for a in actions_to_perform
                    if a["action"] == "delete_s3"
                ),
                None,
            )
            if not dry_run and target_bucket_del_batch:
                deleted_count_batch = 0
                for i in range(0, len(s3_deletions_batch), 1000):
                    batch = s3_deletions_batch[i : i + 1000]
                    delete_payload = {"Objects": batch, "Quiet": False}
                    try:
                        response = self.client.delete_objects(
                            Bucket=target_bucket_del_batch, Delete=delete_payload
                        )
                        deleted_count_batch += len(batch)
                        if "Errors" in response and response["Errors"]:
                            deleted_count_batch -= len(response["Errors"])
                            for error in response["Errors"]:
                                logger.error(
                                    f"ERROR deleting S3 object {error['Key']}: {error['Code']} - {error['Message']}"
                                )
                    except ClientError as e:
                        logger.error(f"ERROR deleting S3 objects batch: {e}")
                        deleted_count_batch = 0
                    except Exception as e:
                        logger.error(f"ERROR deleting S3 objects batch: {e}")
                        deleted_count_batch = 0
                deletions_done += deleted_count_batch
            elif target_bucket_del_batch:
                deletions_done = len(s3_deletions_batch)
            else:
                logger.warning(
                    "Warning: Could not determine target bucket for S3 deletion batch."
                )

        if dry_run:
            if self.verbose:
                upload_count = sum(
                    1 for a in actions_to_perform if a["action"] == "upload"
                )
                download_count = sum(
                    1 for a in actions_to_perform if a["action"] == "download"
                )
                delete_s3_count = len(s3_deletions_batch)
                delete_local_count = sum(
                    1 for a in actions_to_perform if a["action"] == "delete_local"
                )
                logger.info("\n--- DRY RUN SUMMARY ---")
                if sync_direction == "upload":
                    logger.info(f"Would upload: {upload_count} file(s)")
                    if delete:
                        logger.info(
                            f"Would delete from S3: {delete_s3_count} object(s)"
                        )
                elif sync_direction == "download":
                    logger.info(f"Would download: {download_count} file(s)")
                    if delete:
                        logger.info(
                            f"Would delete locally: {delete_local_count} file(s)"
                        )
                logger.info("--- END DRY RUN ---")
        else:
            if self.verbose:
                if sync_direction == "upload":
                    logger.info(
                        f"Sync completed. Uploaded: {uploads_done} file(s). Deleted from S3: {deletions_done if delete else 0} object(s)."
                    )
                elif sync_direction == "download":
                    logger.info(
                        f"Sync completed. Downloaded: {downloads_done} file(s). Deleted locally: {deletions_done if delete else 0} file(s)."
                    )

    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        """
        Copies files or directories between local paths and S3 URIs.
        Handles:
        - Local file to S3 object
        - Local directory to S3 prefix (recursive)
        - S3 object to local file
        - S3 prefix to local directory (recursive)
        Does NOT handle:
        - Local to Local (use shutil)
        - S3 to S3 (use AWS CLI or boto3 object copy)
        Args:
            source (str): The source path (local file/dir or s3://...).
            destination (str): The destination path (local file/dir or s3://...).
        """
        src_bucket, src_prefix = self._parse_path(source)
        dest_bucket, dest_prefix = self._parse_path(destination)

        if src_bucket is None and dest_bucket is None:
            logger.error(
                "Error: Both source and destination are local. Use 'shutil.copy' or 'shutil.copytree'."
            )
            return
        if src_bucket is not None and dest_bucket is not None:
            logger.error(
                "Error: S3 to S3 copy not implemented. Use 'aws s3 cp' or boto3 'copy_object'."
            )
            return

        # Upload: Local to S3
        if src_bucket is None and dest_bucket is not None:
            if not os.path.exists(source):
                logger.error(f"Error: Local source path not found: {source}")
                return
            current_dest_prefix = dest_prefix or ""

            if os.path.isfile(source):
                if not current_dest_prefix or destination.endswith("/"):
                    s3_key = os.path.join(
                        current_dest_prefix, os.path.basename(source)
                    ).replace("\\", "/")
                else:
                    s3_key = current_dest_prefix
                if self.verbose:
                    logger.info(f"Uploading {source} to s3://{dest_bucket}/{s3_key}")
                try:
                    self.client.upload_file(source, dest_bucket, s3_key)
                    if self.verbose:
                        logger.info("Upload complete.")
                except ClientError as e:
                    logger.error(f"ERROR uploading {source}: {e}")
                except Exception as e:
                    logger.error(f"ERROR uploading {source}: {e}")

            elif os.path.isdir(source):
                if self.verbose:
                    logger.info(
                        f"Uploading directory {source}/* to s3://{dest_bucket}/{current_dest_prefix}/"
                    )
                files_uploaded = files_failed = 0
                for root, _, files in os.walk(source):
                    for file in files:
                        local_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_path, source)
                        s3_key = os.path.join(
                            current_dest_prefix, relative_path
                        ).replace("\\", "/")
                        if self.verbose:
                            logger.debug(
                                f"  Uploading {local_path} to s3://{dest_bucket}/{s3_key}"
                            )
                        try:
                            self.client.upload_file(local_path, dest_bucket, s3_key)
                            files_uploaded += 1
                        except ClientError as e:
                            logger.error(f"  ERROR uploading {local_path}: {e}")
                            files_failed += 1
                        except Exception as e:
                            logger.error(f"  ERROR uploading {local_path}: {e}")
                            files_failed += 1
                if self.verbose:
                    logger.info(
                        f"Directory upload complete. Files uploaded: {files_uploaded}, Failed: {files_failed}"
                    )
            else:
                logger.error(
                    f"Error: Source {source} is neither a file nor a directory."
                )

        # Download: S3 to Local
        elif src_bucket is not None and dest_bucket is None:
            is_prefix_download = False
            single_object_key = None
            current_src_prefix = src_prefix or ""  # Ensure not None

            if source.endswith("/"):
                is_prefix_download = True
            else:
                try:
                    if current_src_prefix:
                        self.client.head_object(
                            Bucket=src_bucket, Key=current_src_prefix
                        )
                        single_object_key = current_src_prefix
                    else:
                        # Path like s3://bucket, treat as prefix download
                        is_prefix_download = True
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        is_prefix_download = True  # Assume prefix if object not found
                    elif e.response["Error"]["Code"] == "NoSuchBucket":
                        logger.error(f"Error: Source bucket '{src_bucket}' not found.")
                        return
                    else:
                        logger.error(
                            f"Error checking S3 source s3://{src_bucket}/{current_src_prefix}: {e}"
                        )
                        return
                except Exception as e:
                    logger.error(
                        f"Error checking S3 source s3://{src_bucket}/{current_src_prefix}: {e}"
                    )
                    return

            if single_object_key is not None:
                if os.path.isdir(destination) or destination.endswith(os.sep):
                    local_dest_path = os.path.join(
                        destination, os.path.basename(single_object_key)
                    )
                    os.makedirs(destination, exist_ok=True)
                else:
                    local_dest_path = destination
                    parent_dir = os.path.dirname(local_dest_path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)
                if self.verbose:
                    logger.info(
                        f"Downloading s3://{src_bucket}/{single_object_key} to {local_dest_path}"
                    )
                try:
                    self.client.download_file(
                        src_bucket, single_object_key, local_dest_path
                    )
                    if self.verbose:
                        logger.info("Download complete.")
                except ClientError as e:
                    logger.error(f"ERROR downloading {single_object_key}: {e}")
                except OSError as e:
                    logger.error(f"ERROR creating/writing {local_dest_path}: {e}")
                except Exception as e:
                    logger.error(f"ERROR downloading {single_object_key}: {e}")

            elif is_prefix_download:
                if os.path.exists(destination) and not os.path.isdir(destination):
                    logger.error(
                        f"Error: Local destination '{destination}' exists but is not a directory."
                    )
                    return
                os.makedirs(destination, exist_ok=True)
                if self.verbose:
                    logger.info(
                        f"Downloading prefix s3://{src_bucket}/{current_src_prefix}/* to {destination}/"
                    )
                paginator = self.client.get_paginator("list_objects_v2")
                files_downloaded = files_failed = 0
                operation_parameters = {"Bucket": src_bucket}
                # The problematic line for the linter, re-adding type ignore
                if current_src_prefix:  # type: ignore
                    operation_parameters["Prefix"] = current_src_prefix
                try:
                    page_iterator = paginator.paginate(**operation_parameters)
                    found_objects = False
                    for page in page_iterator:
                        if "Contents" in page:
                            found_objects = True
                            for obj in page["Contents"]:
                                s3_key = obj["Key"]
                                if s3_key.endswith("/") and obj["Size"] == 0:
                                    continue
                                relative_key = s3_key
                                if current_src_prefix:
                                    if s3_key.startswith(current_src_prefix):
                                        if s3_key == current_src_prefix.rstrip("/"):
                                            relative_key = os.path.basename(s3_key)
                                        else:
                                            prefix_adjusted = current_src_prefix + (
                                                ""
                                                if current_src_prefix.endswith("/")
                                                else "/"
                                            )
                                            if s3_key.startswith(prefix_adjusted):
                                                relative_key = s3_key[
                                                    len(prefix_adjusted) :
                                                ]
                                            elif not current_src_prefix.endswith("/"):
                                                relative_key = s3_key[
                                                    len(current_src_prefix) :
                                                ].lstrip("/")
                                if not relative_key:
                                    continue
                                local_dest_path = os.path.join(
                                    destination, relative_key.replace("/", os.sep)
                                )
                                local_dest_dir = os.path.dirname(local_dest_path)
                                if self.verbose:
                                    logger.debug(
                                        f"  Downloading s3://{src_bucket}/{s3_key} to {local_dest_path}"
                                    )
                                try:
                                    if local_dest_dir:
                                        os.makedirs(local_dest_dir, exist_ok=True)
                                    self.client.download_file(
                                        src_bucket, s3_key, local_dest_path
                                    )
                                    files_downloaded += 1
                                except ClientError as e:
                                    logger.error(f"  ERROR downloading {s3_key}: {e}")
                                    files_failed += 1
                                except OSError as e:
                                    logger.error(
                                        f"  ERROR creating/writing {local_dest_path}: {e}"
                                    )
                                    files_failed += 1
                                except Exception as e:
                                    logger.error(f"  ERROR downloading {s3_key}: {e}")
                                    files_failed += 1
                    if not found_objects and self.verbose:
                        logger.warning(
                            f"Warning: No objects found at source prefix s3://{src_bucket}/{current_src_prefix}"
                        )
                    if self.verbose:
                        logger.info(
                            f"Prefix download complete. Files downloaded: {files_downloaded}, Failed: {files_failed}"
                        )
                except ClientError as e:
                    if e.response["Error"]["Code"] == "NoSuchBucket":
                        logger.error(f"Error: Source bucket '{src_bucket}' not found.")
                    else:
                        logger.error(
                            f"Error listing objects in s3://{src_bucket}/{current_src_prefix}: {e}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error listing objects in s3://{src_bucket}/{current_src_prefix}: {e}"
                    )
        else:
            logger.error("Error: Unknown copy operation type.")

    def check(self, path_uri: str) -> bool:
        """
        Check if an object or prefix exists in an S3 bucket using an S3 URI.

        Args:
            path_uri (str): The S3 URI (e.g., 's3://my-bucket/my-key' or 's3://my-bucket/my-prefix/').
                          Use a trailing '/' to check for a prefix/directory.

        Returns:
            bool: True if the object or prefix exists, False otherwise.
        """
        # Use the class client and parse method
        bucket_name, s3_key = self._parse_path(path_uri)

        if bucket_name is None or s3_key is None:
            # _parse_path returns None, None if scheme is not 's3'
            logger.error(f"Error: Invalid S3 URI format: {path_uri}")
            return False

        is_prefix = s3_key.endswith("/")

        try:
            if is_prefix:
                # Check for prefix existence by listing objects
                # Handle the case where s3_key might be empty if URI is just s3://bucket/
                list_prefix = s3_key if s3_key else ""
                response = self.client.list_objects_v2(
                    Bucket=bucket_name, Prefix=list_prefix, MaxKeys=1
                )
                # Check if any objects OR common prefixes (folders) are returned for the prefix
                return "Contents" in response or "CommonPrefixes" in response
            else:
                # Check for object existence
                self.client.head_object(Bucket=bucket_name, Key=s3_key)
                return True
        except ClientError as e:  # Catch boto3 ClientError first
            # If head_object returns 404 (NoSuchKey), the object doesn't exist
            # list_objects_v2 does not raise NoSuchKey for prefixes
            if e.response["Error"]["Code"] == "404":
                return False
            elif e.response["Error"]["Code"] == "NoSuchBucket":
                if self.verbose:
                    logger.error(
                        f"Error: Bucket '{bucket_name}' not found (from URI: {path_uri})."
                    )
                return False
            # Handle other potential errors like AccessDenied differently if needed
            logger.error(f"Error checking {path_uri}: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred checking {path_uri}: {e}")
            return False


# Standalone helper for RcloneBucket._is_rclone_path
def _is_rclone_path_standalone(path: str) -> bool:
    """
    Standalone helper: Determines if a path string is an rclone path (e.g., "remote:path",
    ":backend:path", or "s3://bucket/path").
    """
    parsed_url = urlparse(path)

    # Explicitly allow s3:// paths as rclone paths
    if parsed_url.scheme == "s3" and parsed_url.netloc:
        return True

    # Check for Windows drive letter paths (e.g., C:\path or C:) - these are local.
    if os.name == "nt":
        if len(path) >= 2 and path[0].isalpha() and path[1] == ":":
            if len(path) == 2:  # e.g., "C:"
                return False  # Local current directory on drive
            if path[2] in ["\\", "/"]:  # e.g., "C:\foo" or "C:/foo"
                return False  # Local absolute path

    # Handle file:// scheme as local
    if parsed_url.scheme == "file":
        return False

    # If it has another scheme (e.g., http, ftp) and a network location,
    # it's a URL, not typically an rclone path for copy/sync operations in this context.
    if parsed_url.scheme and parsed_url.scheme != "s3" and parsed_url.netloc:
        return False

    # If the path contains a colon, it's likely an rclone remote path
    # (e.g., "myremote:path" or ":s3:path" or "s3:path" if scheme not picked up by urlparse for s3:).
    # This check comes after specific local/URL patterns are ruled out.
    if ":" in path:
        return True

    # Default to local if none of the above (e.g., "/abs/path", "rel/path")
    return False


class RcloneBucket(StorageBucket):
    """Handles interactions with storage using the rclone-python library."""

    def __init__(
        self,
        verbose: bool = True,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """
        Initializes the RcloneBucket handler.

        Args:
            verbose (bool): If True, prints status messages and sets rclone-python log level.
            aws_access_key_id (Optional[str]): AWS Access Key ID for rclone S3 remotes.
            aws_secret_access_key (Optional[str]): AWS Secret Access Key for rclone S3 remotes.
            aws_session_token (Optional[str]): AWS Session Token for rclone S3 remotes.
            region (str): AWS region for S3 operations. Defaults to "us-east-1".
        """
        super().__init__(verbose=verbose)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.region = region

        if not is_rclone_installed():
            logger.error(
                "rclone command not found. Please ensure rclone is installed and configured correctly (https://rclone.org/install/)."
            )
            # Consider raising an exception if rclone CLI is essential for rclone-python to function.
            return

        if self.verbose:
            logger.info("Initialized RcloneBucket with rclone-python.")
            logger.info("rclone-python log level set to DEBUG.")
        else:
            logger.info("rclone-python log level set to WARNING.")

        # Store a mtime tolerance, similar to S3 Bucket
        self.mtime_tolerance = timedelta(seconds=2)

    def _is_rclone_path(self, path: str) -> bool:
        """
        Determines if a path string is an rclone path (e.g., "remote:path").
        """
        return _is_rclone_path_standalone(path)

    def _execute_with_aws_env(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Helper to execute rclone functions with temporary AWS env vars if provided."""
        old_env: Dict[str, Optional[str]] = {}
        aws_vars = {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            "AWS_SESSION_TOKEN": self.aws_session_token,
            # Add rclone-specific S3 configuration
            "RCLONE_CONFIG_S3_TYPE": "s3",
            "RCLONE_CONFIG_S3_PROVIDER": "AWS",
            "RCLONE_CONFIG_S3_ENV_AUTH": "true",
            "RCLONE_CONFIG_S3_REGION": self.region,
        }

        try:
            for key, value in aws_vars.items():
                if value is not None:
                    old_env[key] = os.environ.get(key)
                    os.environ[key] = value
                elif key in os.environ:  # Value is None but was set in env
                    old_env[key] = os.environ.get(key)
                    del os.environ[key]

            # Ensure stderr is captured by setting show_progress to False
            if "show_progress" not in kwargs:
                kwargs["show_progress"] = False

            # Set DEBUG log level for rclone to get more verbose output
            old_log_level = logging.getLogger("rclone").level
            logging.getLogger("rclone").setLevel(logging.DEBUG)

            # Convert s3:// URLs if needed
            modified_args = list(args)
            for i, arg in enumerate(modified_args):
                if isinstance(arg, str) and arg.startswith("s3://"):
                    # Convert s3://bucket/path to s3:bucket/path
                    arg_parts = arg[5:].split("/", 1)
                    bucket_name = arg_parts[0]
                    path = arg_parts[1] if len(arg_parts) > 1 else ""
                    modified_args[i] = f"s3:{bucket_name}/{path}"

            try:
                return func(*modified_args, **kwargs)
            finally:
                # Restore the original log level
                logging.getLogger("rclone").setLevel(old_log_level)
        finally:
            for key, value in old_env.items():
                if value is None:
                    if key in os.environ:  # It was set by us, now remove
                        del os.environ[key]
                else:
                    os.environ[key] = value

            # Clean up any env vars we set but weren't in old_env
            for key in aws_vars.keys():
                if key not in old_env and key in os.environ:
                    del os.environ[key]

    def sync(
        self,
        source: str,
        destination: str,
        delete: bool = False,
        dry_run: bool = False,
        excludes: Optional[List[str]] = None,
    ) -> None:
        if not is_rclone_installed():
            logger.error("Cannot sync: rclone command not found.")
            return

        if self.verbose:
            logger.info(f"Rclone sync: {source} -> {destination}")
            if delete:
                logger.info("Deletion enabled.")
            if dry_run:
                logger.info("Dry run mode.")
            if excludes:
                logger.info(f"Excludes: {excludes}")

        rc_args = [
            "--modify-window=2s",
            "--log-level=DEBUG" if self.verbose else "--log-level=INFO",
            "--log-format=date,time,level,message",
            # "--progress",  # Add progress display
        ]
        if dry_run:
            rc_args.append("--dry-run")
        if delete:
            rc_args.append("--delete-after")

        if excludes:
            for ex_pattern in excludes:
                rc_args.append(f"--exclude={ex_pattern}")

        # Set environment variables for AWS credentials if they exist
        env = os.environ.copy()
        if self.aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_session_token:
            env["AWS_SESSION_TOKEN"] = self.aws_session_token

        # Set rclone-specific environment variables
        env["RCLONE_CONFIG_S3_TYPE"] = "s3"
        env["RCLONE_CONFIG_S3_PROVIDER"] = "AWS"
        env["RCLONE_CONFIG_S3_ENV_AUTH"] = "true"
        env["RCLONE_CONFIG_S3_REGION"] = self.region

        # Convert s3:// URLs if needed
        rclone_src = source
        rclone_dest = destination

        # If source or destination uses s3:// URL format, convert it for rclone CLI
        if source.startswith("s3://"):
            # Convert s3://bucket/path to s3:bucket/path
            source_parts = source[5:].split("/", 1)
            bucket_name = source_parts[0]
            path = source_parts[1] if len(source_parts) > 1 else ""
            rclone_src = f"s3:{bucket_name}/{path}"

        if destination.startswith("s3://"):
            # Convert s3://bucket/path to s3:bucket/path
            destination_parts = destination[5:].split("/", 1)
            bucket_name = destination_parts[0]
            path = destination_parts[1] if len(destination_parts) > 1 else ""
            rclone_dest = f"s3:{bucket_name}/{path}"

        # Build the rclone command
        cmd = ["rclone", "sync", rclone_src, rclone_dest] + rc_args

        if self.verbose:
            logger.info(f"Running command: {' '.join(cmd)}")

        try:
            # Run the command and capture output
            process = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            # Always log the stdout and stderr for verbose mode
            if self.verbose:
                if process.stdout:
                    logger.info(f"Rclone stdout:\n{process.stdout}")
                if process.stderr:
                    logger.info(f"Rclone stderr:\n{process.stderr}")

            if process.returncode == 0:
                logger.info(
                    f"Rclone sync completed successfully from {source} to {destination}."
                )
                if dry_run:
                    logger.info(
                        "Dry run summary (see rclone output above for details)."
                    )
            else:
                logger.error(f"Rclone sync failed with exit code: {process.returncode}")

                if (
                    not self.verbose
                ):  # Only log these again if not already logged in verbose mode
                    if process.stdout:
                        logger.error(f"Rclone stdout:\n{process.stdout}")
                    if process.stderr:
                        logger.error(f"Rclone stderr:\n{process.stderr}")
        except Exception as e:
            logger.error(f"Error running rclone sync command: {e}")

    def copy(
        self,
        source: str,
        destination: str,
    ) -> None:
        if not is_rclone_installed():
            logger.error("Cannot copy: rclone command not found.")
            return

        # Determine if source/destination are rclone paths or local
        is_src_rclone = self._is_rclone_path(source)
        is_dest_rclone = self._is_rclone_path(destination)

        if not is_src_rclone and not is_dest_rclone:
            logger.error(
                "Error: Both source and destination are local. Use 'shutil.copy' or 'shutil.copytree'."
            )
            return

        if self.verbose:
            logger.info(f"Rclone copy: {source} -> {destination}")

        rc_args = [
            "--log-level=DEBUG" if self.verbose else "--log-level=INFO",
            "--log-format=date,time,level,message",
            # "--progress",  # Add progress display
        ]

        # Set environment variables for AWS credentials if they exist
        env = os.environ.copy()
        if self.aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_session_token:
            env["AWS_SESSION_TOKEN"] = self.aws_session_token

        # Set rclone-specific environment variables
        env["RCLONE_CONFIG_S3_TYPE"] = "s3"
        env["RCLONE_CONFIG_S3_PROVIDER"] = "AWS"
        env["RCLONE_CONFIG_S3_ENV_AUTH"] = "true"
        env["RCLONE_CONFIG_S3_REGION"] = self.region

        # Convert s3:// URLs if needed
        rclone_src = source
        rclone_dest = destination

        # If source or destination uses s3:// URL format, convert it for rclone CLI
        if source.startswith("s3://"):
            # Convert s3://bucket/path to s3:bucket/path
            source_parts = source[5:].split("/", 1)
            bucket_name = source_parts[0]
            path = source_parts[1] if len(source_parts) > 1 else ""
            rclone_src = f"s3:{bucket_name}/{path}"

        if destination.startswith("s3://"):
            # Convert s3://bucket/path to s3:bucket/path
            destination_parts = destination[5:].split("/", 1)
            bucket_name = destination_parts[0]
            path = destination_parts[1] if len(destination_parts) > 1 else ""
            rclone_dest = f"s3:{bucket_name}/{path}"

        # Build the rclone command
        cmd = ["rclone", "copy", rclone_src, rclone_dest] + rc_args

        if self.verbose:
            logger.info(f"Running command: {' '.join(cmd)}")

        try:
            # Run the command and capture output
            process = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            # Always log the stdout and stderr for verbose mode
            if self.verbose:
                if process.stdout:
                    logger.info(f"Rclone stdout:\n{process.stdout}")
                if process.stderr:
                    logger.info(f"Rclone stderr:\n{process.stderr}")

            if process.returncode == 0:
                logger.info(
                    f"Rclone copy completed successfully from {source} to {destination}."
                )
            else:
                logger.error(f"Rclone copy failed with exit code: {process.returncode}")

                if (
                    not self.verbose
                ):  # Only log these again if not already logged in verbose mode
                    if process.stdout:
                        logger.error(f"Rclone stdout:\n{process.stdout}")
                    if process.stderr:
                        logger.error(f"Rclone stderr:\n{process.stderr}")
        except Exception as e:
            logger.error(f"Error running rclone copy command: {e}")

    def check(self, path_uri: str) -> bool:
        if not is_rclone_installed():
            logger.error("Cannot check path: rclone command not found.")
            return False

        if self.verbose:
            logger.info(f"Checking existence of path: {path_uri}")

        # Convert s3:// URL if needed
        rclone_path = path_uri
        if path_uri.startswith("s3://"):
            # Convert s3://bucket/path to s3:bucket/path
            path_parts = path_uri[5:].split("/", 1)
            bucket_name = path_parts[0]
            path = path_parts[1] if len(path_parts) > 1 else ""
            rclone_path = f"s3:{bucket_name}/{path}"

        # Set environment variables for AWS credentials if they exist
        env = os.environ.copy()
        if self.aws_access_key_id:
            env["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            env["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_session_token:
            env["AWS_SESSION_TOKEN"] = self.aws_session_token

        # Set rclone-specific environment variables
        env["RCLONE_CONFIG_S3_TYPE"] = "s3"
        env["RCLONE_CONFIG_S3_PROVIDER"] = "AWS"
        env["RCLONE_CONFIG_S3_ENV_AUTH"] = "true"
        env["RCLONE_CONFIG_S3_REGION"] = self.region

        # Build the rclone command (size is a good way to check existence)
        cmd = ["rclone", "size", rclone_path, "--json"]

        try:
            # Run the command and capture output
            process = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if process.returncode == 0:
                if self.verbose:
                    logger.debug(f"Path {path_uri} exists and is accessible.")
                return True
            else:
                # Check if this is a "not found" error vs. another type of error
                err_lower = process.stderr.lower() if process.stderr else ""
                not_found_indicators = [
                    "directory not found",
                    "object not found",
                    "path not found",
                    "no such file or directory",
                    "source or destination not found",
                    "error: couldn't find file",
                    "failed to size: can't find object",
                    "can't find source directory",
                    "source not found",
                ]
                is_not_found = any(
                    indicator in err_lower for indicator in not_found_indicators
                )

                if is_not_found:
                    if self.verbose:
                        logger.debug(f"Path {path_uri} not found.")
                    return False
                else:
                    # Other type of error (permissions, network, etc.)
                    logger.warning(f"Error checking path {path_uri}: {process.stderr}")
                    return False
        except Exception as e:
            logger.error(f"Error running rclone check command: {e}")
            return False


# Factory function Bucket()
def Bucket(
    verbose: bool = True,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region: str = "us-east-1",
) -> StorageBucket:
    """
    Factory function to create a bucket instance.
    Returns RcloneBucket if rclone is available and installed, otherwise S3Bucket.

    Args:
        verbose (bool): If True, prints status messages. Defaults to True.
        aws_access_key_id (Optional[str]): Temporary AWS Access Key ID (for S3Bucket).
        aws_secret_access_key (Optional[str]): Temporary AWS Secret Access Key (for S3Bucket).
        aws_session_token (Optional[str]): Temporary AWS Session Token (for S3Bucket).
        region (str): AWS region for S3 operations. Defaults to "us-east-1".

    Returns:
        StorageBucket: An instance of RcloneBucket or S3Bucket.
    """
    if is_rclone_installed():
        print("rclone is installed and available")
        if verbose:
            logger.info("rclone is installed and available. Using RcloneBucket.")
        return RcloneBucket(
            verbose=verbose,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region=region,
        )
    else:
        print("rclone not installed or not available")
        if verbose:
            logger.info(
                "rclone not installed or not available. Falling back to S3Bucket."
            )
        return S3Bucket(
            verbose=verbose,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region=region,
        )
