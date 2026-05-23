
import sys
sys.path.append('./src/agent')  # Add agent directory to path for imports

from langchain_agent import main, lf_run, fuse_client, conversation_id

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\nInterrupted. Finishing Langfuse run if present...")
		try:
			if lf_run is not None:
				if hasattr(lf_run, 'finish'):
					lf_run.finish()
				elif fuse_client is not None and hasattr(fuse_client, 'finish_run'):
					fuse_client.finish_run(conversation_id)
			if fuse_client is not None:
				fuse_client.flush()
		except Exception:
			pass
