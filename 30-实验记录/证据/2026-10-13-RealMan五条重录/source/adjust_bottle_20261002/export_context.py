resume_context={'args':args,'ready':ready,'grasp_setting':globals().get('grasp_setting'),'accepted_grasp_setting':globals().get('accepted_grasp_setting'),'accepted_seed':globals().get('accepted_seed'),'current_output':str(OUT)}
(BASE/'resume_context.json').write_text(json.dumps(resume_context,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
print('CONTEXT_EXPORTED',str(OUT),flush=True)
