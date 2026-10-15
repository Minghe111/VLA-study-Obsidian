"""Task-scoped diagnostics only; no scene, timestep or controller writes."""
import faulthandler,os,signal,sys
if os.environ.get('REALMAN_STAGE_DIAGNOSTIC')=='1':
    faulthandler.enable(file=sys.stderr,all_threads=True)
    faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=True,chain=False)
    faulthandler.dump_traceback_later(90,repeat=True,file=sys.stderr,exit=False)
    print('TASK_STACK_DIAGNOSTIC_ENABLED pid='+str(os.getpid()),file=sys.stderr,flush=True)
