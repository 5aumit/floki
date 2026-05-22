"""
Langfuse tracing setup utilities for MLflow agent
"""

import os
import time
import logging

# Optionally import langfuse if available
try:
    from langfuse import get_client, propagate_attributes
    from langfuse.langchain import CallbackHandler
except ImportError:
    get_client = propagate_attributes = CallbackHandler = None
    logging.warning("Langfuse not installed; tracing will be disabled.")

# Exposed module-level variable for user metadata
langfuse_user = None


def setup_langfuse(config):
    """
    Set up Langfuse tracing based on config and environment variables.
    Returns: langfuse_handler, fuse_client, conversation_id, lf_run, FLUSH_PER_QUERY
    """
    global langfuse_user
    langfuse_handler = None
    langfuse_user = config.get('langfuse', {}).get('user', 'unknown_user')
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    host = os.getenv('LANGFUSE_BASE_URL')
    fuse_client = None
    conversation_id = None
    lf_run = None
    FLUSH_PER_QUERY = os.getenv('LANGFUSE_FLUSH_PER_QUERY', 'false').lower() == 'true'
    if get_client is None:
        return None, None, None, None, None, FLUSH_PER_QUERY
    try:
        if public_key:
            try:
                fuse_client = get_client(public_key=public_key, host=host) if host else get_client(public_key=public_key)
            except TypeError:
                os.environ.setdefault('LANGFUSE_PUBLIC_KEY', public_key)
                if host:
                    os.environ.setdefault('LANGFUSE_BASE_URL', host)
                fuse_client = get_client()
            try:
                langfuse_handler = CallbackHandler(client=fuse_client)
            except TypeError:
                langfuse_handler = CallbackHandler()
            try:
                import uuid
                conversation_id = f"cli-{uuid.uuid4().hex[:8]}"
            except Exception:
                conversation_id = f"cli-{int(time.time())}"
            try:
                run_kwargs = {}
                if langfuse_user:
                    run_kwargs['user'] = langfuse_user
                if hasattr(fuse_client, 'start_run'):
                    lf_run = fuse_client.start_run(name=conversation_id, **run_kwargs)
                elif hasattr(fuse_client, 'runs') and hasattr(fuse_client.runs, 'create'):
                    lf_run = fuse_client.runs.create(name=conversation_id, **run_kwargs)
            except Exception:
                lf_run = None
            logging.info("Langfuse handler initialized. conversation_id=%s", conversation_id)
        else:
            logging.info("LANGFUSE_PUBLIC_KEY not set; Langfuse tracing disabled.")
    except Exception as e:
        logging.exception("Failed to initialize Langfuse handler: %s", e)
        langfuse_handler = None

    # Return the initialized objects (may be None if disabled)
    return langfuse_handler, fuse_client, conversation_id, lf_run, langfuse_user, FLUSH_PER_QUERY
# Re-export propagate_attributes for convenience
__all__ = ["setup_langfuse", "propagate_attributes", "langfuse_user"]
