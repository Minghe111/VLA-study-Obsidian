from support_departure_certificate_v20 import certify_support_departure
import preflight_place_goals as ppg20
snapshot20=ppg20._snapshot(task)
failed20=json.loads((BASE/'closed_lift_v20_readonly.json').read_text())
line20=failed20['planning_metadata']['RRT_attempts'][0]['retiming']['waypoints']
cert20=certify_support_departure(task,np.asarray(line20),'left',palm_to_bottle=np.asarray(grip_measure(task)['relative_bottle_palm']),endpoint_translation_tolerance_m=5e-5)
ppg20._assert_snapshot(task,snapshot20)
(BASE/'closed_lift_v20_certificate_readonly.json').write_text(json.dumps(cert20,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),indent=2))
print('CERT20',cert20['passed'],cert20['signed_support_departure_certificate'],flush=True)
